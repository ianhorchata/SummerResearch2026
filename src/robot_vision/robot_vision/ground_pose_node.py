"""2D detections -> ground-plane poses (single or stereo-validated).

Services
--------
``/vision/detect_pose`` (DetectObjectPose)
    Pick one Detection2D and project its bbox center to the floor.

``/vision/detect_poses`` (DetectObjectPoses)
    Project every detection (numeric ID order). With ``stereo:=true``, run
    left and right, keep only objects whose ground poses agree within
    ``match_tol_m``, then triangulate each match from L/R bbox centroids
    + camera intrinsics (ray midpoint in ``target_frame``). Also returns
    apparent bbox ``widths_m`` / ``heights_m`` from depth + intrinsics.

Pipeline (per camera)
---------------------
1. Call ``/vision/detect``.
2. Back-project each bbox center through camera intrinsics.
3. Transform the ray into ``target_frame`` (default ``base_link``).
4. Intersect with the horizontal plane z = ground_z.

Stereo (``stereo:=true``)
-------------------------
1. Detect + ground-project left and right (for association only).
2. Greedy-match by ground XY within ``match_tol_m``.
3. Triangulate each pair from pixel centroids through K and TF
   (works with toe-in; not a parallel-stereo disparity formula).

    ros2 service call /vision/detect_poses robot_interfaces/srv/DetectObjectPoses \\
      "{stereo: true, match_tol_m: 0.15}"
    ros2 service call /vision/detect_poses robot_interfaces/srv/DetectObjectPoses \\
      "{stereo: false, camera: 'left'}"
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion, Vector3
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401  # registers PoseStamped <-> TF types

from robot_interfaces.srv import DetectObjectPose, DetectObjectPoses, DetectObjects


class GroundPoseNode(Node):
    def __init__(self):
        super().__init__("ground_pose_node")

        self.declare_parameter("default_camera", "left")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("tf_timeout_sec", 0.5)
        self.declare_parameter("match_tol_m", 0.15)
        # Prefer real camera_info once calibrated; set true only as a fallback.
        self.declare_parameter("use_approx_intrinsics", False)
        self.declare_parameter("approx_hfov_deg", 62.0)  # tune to your CSI lens
        self.declare_parameter("approx_image_width", 1280)
        self.declare_parameter("approx_image_height", 720)

        self._default_camera = self.get_parameter("default_camera").value
        self._default_target = self.get_parameter("target_frame").value
        self._default_ground_z = float(self.get_parameter("ground_z").value)
        self._tf_timeout = float(self.get_parameter("tf_timeout_sec").value)
        self._default_match_tol = float(self.get_parameter("match_tol_m").value)
        self._use_approx_k = self._as_bool(
            self.get_parameter("use_approx_intrinsics").value)
        self._approx_hfov = math.radians(
            float(self.get_parameter("approx_hfov_deg").value))
        self._approx_w = int(self.get_parameter("approx_image_width").value)
        self._approx_h = int(self.get_parameter("approx_image_height").value)

        self._cb_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._camera_info: dict[str, CameraInfo] = {}
        for cam in ("left", "right"):
            self.create_subscription(
                CameraInfo,
                f"{cam}/camera_info",
                self._make_info_cb(cam),
                sensor_qos,
                callback_group=self._cb_group,
            )

        self._detect_cli = self.create_client(
            DetectObjects, "vision/detect", callback_group=self._cb_group)

        self._pose_pub = self.create_publisher(PoseStamped, "vision/object_pose", 10)
        self._poses_pub = self.create_publisher(PoseArray, "vision/object_poses", 10)

        self.create_service(
            DetectObjectPose,
            "vision/detect_pose",
            self._on_detect_pose,
            callback_group=self._cb_group,
        )
        self.create_service(
            DetectObjectPoses,
            "vision/detect_poses",
            self._on_detect_poses,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            "ground_pose_node ready; /vision/detect_pose (single) and "
            f"/vision/detect_poses (all / stereo, match_tol={self._default_match_tol:.2f}m, "
            f"target_frame={self._default_target}, ground_z={self._default_ground_z})"
        )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _make_info_cb(self, cam: str):
        def _cb(msg: CameraInfo):
            self._camera_info[cam] = msg
        return _cb

    def _resolve_ground_z(self, request_ground_z: float) -> float:
        # Nonzero request wins; otherwise use the node param (default 0 = floor).
        if abs(float(request_ground_z)) > 0.0:
            return float(request_ground_z)
        return self._default_ground_z

    # ------------------------------------------------------------------ detect

    def _call_detect(self, cam: str, confidence: float, timeout_sec: float = 60.0):
        """Call /vision/detect. Returns (ok, message, Detection2DArray|None).

        Polls the future instead of ``result(timeout=...)`` — rclpy's Future
        does not accept a timeout kwarg. MultiThreadedExecutor delivers the
        response on another thread while we wait (do not spin here).
        """
        if not self._detect_cli.wait_for_service(timeout_sec=2.0):
            return False, "/vision/detect is not available (is vision_node up?)", None

        det_req = DetectObjects.Request()
        det_req.camera = cam
        det_req.confidence = confidence
        future = self._detect_cli.call_async(det_req)
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() >= deadline:
                return (
                    False,
                    f"Timed out waiting for /vision/detect ({cam}, {timeout_sec:.0f}s)",
                    None,
                )
            time.sleep(0.02)
        try:
            det_res = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, f"Failed waiting for /vision/detect: {exc}", None
        if det_res is None:
            return False, "Empty response from /vision/detect", None
        if not det_res.success:
            return False, f"vision/detect failed: {det_res.message}", det_res.detections
        return True, det_res.message, det_res.detections

    # ----------------------------------------------------------- single pose

    def _on_detect_pose(self, request, response):
        cam = request.camera or self._default_camera
        target = request.target_frame or self._default_target
        ground_z = self._resolve_ground_z(request.ground_z)

        ok, msg, dets = self._call_detect(cam, request.confidence)
        if dets is not None:
            response.all_detections = dets
        if not ok or dets is None:
            response.success = False
            response.message = msg
            return response

        chosen = self._pick_detection(
            dets,
            class_filter=request.class_filter or "",
            index=int(request.detection_index),
        )
        if chosen is None:
            response.success = False
            response.message = (
                f"No detection to project "
                f"(filter='{request.class_filter}', index={request.detection_index}, "
                f"count={len(dets.detections)})"
            )
            return response

        try:
            pose = self._detection_to_ground_pose(
                cam, chosen, target, ground_z)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Projection failed: {exc}"
            return response

        response.detection = chosen
        response.pose = pose
        response.success = True
        class_id = (
            chosen.results[0].hypothesis.class_id if chosen.results else "?"
        )
        label = f"#{chosen.id}:{class_id}" if chosen.id else class_id
        u = chosen.bbox.center.position.x
        v = chosen.bbox.center.position.y
        response.message = (
            f"Projected '{label}' from '{cam}' pixel ({u:.1f},{v:.1f}) -> "
            f"{target} ({pose.pose.position.x:.3f}, "
            f"{pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})"
        )
        self.get_logger().info(response.message)
        self._pose_pub.publish(pose)
        return response

    # ------------------------------------------------------------- multi pose

    def _on_detect_poses(self, request, response):
        target = request.target_frame or self._default_target
        ground_z = self._resolve_ground_z(request.ground_z)
        match_tol = (
            float(request.match_tol_m)
            if float(request.match_tol_m) > 0.0
            else self._default_match_tol
        )
        stereo = self._as_bool(request.stereo)

        if stereo:
            return self._on_detect_poses_stereo(
                request, response, target, ground_z, match_tol)

        cam = request.camera or self._default_camera
        ok, msg, dets = self._call_detect(cam, request.confidence)
        empty = Detection2DArray()
        response.left_detections = dets if dets is not None else empty
        response.right_detections = empty
        if not ok or dets is None:
            response.success = False
            response.message = msg
            return response

        projected, errors = self._project_detections(
            cam, dets, target, ground_z)
        # Sort by numeric id so order matches annotated #0, #1, ...
        projected.sort(key=lambda item: self._id_sort_key(item[0]))

        out_dets = Detection2DArray()
        out_dets.header = dets.header
        poses: List[PoseStamped] = []
        widths: List[float] = []
        heights: List[float] = []
        for det, pose in projected:
            out_dets.detections.append(det)
            poses.append(pose)
            w_m, h_m = self._bbox_size_m(cam, det, pose)
            widths.append(w_m)
            heights.append(h_m)

        response.detections = out_dets
        response.poses = poses
        response.widths_m = widths
        response.heights_m = heights
        response.match_errors_m = []
        response.success = True
        lines = [
            f"Projected {len(poses)} object(s) from '{cam}' -> {target}",
        ]
        if errors:
            lines.append(f"  skipped {len(errors)} (projection failed)")
        for (det, pose), w_m, h_m in zip(projected, widths, heights):
            lines.append(
                self._format_object_line(
                    det, pose, cam=cam,
                    extra=f"size_m=({w_m:.3f}x{h_m:.3f})",
                )
            )
        response.message = "\n".join(lines)
        for line in lines:
            self.get_logger().info(line)
        self._publish_pose_array(poses, target)
        if poses:
            self._pose_pub.publish(poses[0])
        return response

    def _on_detect_poses_stereo(
        self, request, response, target: str, ground_z: float, match_tol: float
    ):
        ok_l, msg_l, dets_l = self._call_detect("left", request.confidence)
        response.left_detections = dets_l if dets_l is not None else Detection2DArray()
        if not ok_l or dets_l is None:
            response.success = False
            response.message = f"left: {msg_l}"
            response.right_detections = Detection2DArray()
            return response

        ok_r, msg_r, dets_r = self._call_detect("right", request.confidence)
        response.right_detections = dets_r if dets_r is not None else Detection2DArray()
        if not ok_r or dets_r is None:
            response.success = False
            response.message = f"right: {msg_r}"
            return response

        left_proj, _ = self._project_detections(
            "left", dets_l, target, ground_z)
        right_proj, _ = self._project_detections(
            "right", dets_r, target, ground_z)
        left_proj.sort(key=lambda item: self._id_sort_key(item[0]))

        matches = self._match_by_ground_pose(left_proj, right_proj, match_tol)

        out_dets = Detection2DArray()
        out_dets.header = dets_l.header
        poses: List[PoseStamped] = []
        widths: List[float] = []
        heights: List[float] = []
        errors: List[float] = []
        lines = [
            f"Stereo triangulated {len(matches)}/{len(left_proj)} left object(s) "
            f"(right had {len(right_proj)}; tol={match_tol:.2f}m) -> {target}",
        ]
        for left_det, left_pose, right_det, right_pose, err in matches:
            lid = left_det.id if left_det.id else "?"
            rid = right_det.id if right_det.id else "?"
            try:
                tri_pose, tri_res = self._triangulate_detections(
                    left_det, right_det, target)
                used = tri_pose
                range_m = math.hypot(
                    tri_pose.pose.position.x, tri_pose.pose.position.y)
                extra = (
                    f"match=left#{lid}<->right#{rid}  "
                    f"match_err={err:.3f}m  "
                    f"tri_residual={tri_res:.3f}m  "
                    f"range_xy={range_m:.3f}m"
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"Stereo triangulate failed left#{lid}<->right#{rid}: {exc}; "
                    "falling back to averaged ground pose"
                )
                used = self._average_poses(left_pose, right_pose)
                extra = (
                    f"match=left#{lid}<->right#{rid}  "
                    f"match_err={err:.3f}m  "
                    f"tri=fallback_avg_ground"
                )
            w_m, h_m = self._bbox_size_m("left", left_det, used)
            out_dets.detections.append(left_det)
            poses.append(used)
            widths.append(w_m)
            heights.append(h_m)
            errors.append(float(err))
            lines.append(
                self._format_object_line(
                    left_det, used, cam="stereo",
                    extra=f"{extra}  size_m=({w_m:.3f}x{h_m:.3f})",
                )
            )

        response.detections = out_dets
        response.poses = poses
        response.widths_m = widths
        response.heights_m = heights
        response.match_errors_m = errors
        response.success = True
        response.message = "\n".join(lines)
        for line in lines:
            self.get_logger().info(line)
        self._publish_pose_array(poses, target)
        if poses:
            self._pose_pub.publish(poses[0])
        return response

    @staticmethod
    def _format_object_line(
        det: Detection2D,
        pose: PoseStamped,
        cam: str = "",
        extra: str = "",
    ) -> str:
        """One readable debug line: camera, id, bbox size/center, ground pose."""
        oid = det.id if det.id else "?"
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        sx = det.bbox.size_x
        sy = det.bbox.size_y
        px = pose.pose.position.x
        py = pose.pose.position.y
        pz = pose.pose.position.z
        score = max(
            (r.hypothesis.score for r in det.results), default=float("nan"))
        cam_tag = f"cam={cam}  " if cam else ""
        tail = f"  {extra}" if extra else ""
        return (
            f"  {cam_tag}id={oid}  score={score:.2f}  "
            f"center_px=({cx:.1f},{cy:.1f})  size_px=({sx:.1f}x{sy:.1f})  "
            f"pose_{pose.header.frame_id}=(x={px:.3f}, y={py:.3f}, z={pz:.3f})m"
            f"{tail}"
        )

    def _project_detections(
        self,
        cam: str,
        dets: Detection2DArray,
        target: str,
        ground_z: float,
    ) -> Tuple[List[Tuple[Detection2D, PoseStamped]], List[str]]:
        """Project each detection; skip failures. Returns (pairs, error msgs)."""
        pairs: List[Tuple[Detection2D, PoseStamped]] = []
        errors: List[str] = []
        for det in dets.detections:
            try:
                pose = self._detection_to_ground_pose(
                    cam, det, target, ground_z)
                pairs.append((det, pose))
            except Exception as exc:  # noqa: BLE001
                label = det.id or "?"
                errors.append(f"#{label}: {exc}")
                self.get_logger().warn(
                    f"Skip projection cam={cam} id={label}: {exc}"
                )
        return pairs, errors

    @staticmethod
    def _id_sort_key(det: Detection2D) -> Tuple[int, str]:
        try:
            return (int(det.id), det.id)
        except (TypeError, ValueError):
            return (10**9, det.id or "")

    @staticmethod
    def _xy(pose: PoseStamped) -> Tuple[float, float]:
        return pose.pose.position.x, pose.pose.position.y

    @staticmethod
    def _dist_xy(a: PoseStamped, b: PoseStamped) -> float:
        ax, ay = a.pose.position.x, a.pose.position.y
        bx, by = b.pose.position.x, b.pose.position.y
        return math.hypot(ax - bx, ay - by)

    def _match_by_ground_pose(
        self,
        left: List[Tuple[Detection2D, PoseStamped]],
        right: List[Tuple[Detection2D, PoseStamped]],
        tol_m: float,
    ) -> List[Tuple[Detection2D, PoseStamped, Detection2D, PoseStamped, float]]:
        """Greedy nearest-neighbor match left->right within tol_m (left ID order)."""
        used_right: set[int] = set()
        matches: List[
            Tuple[Detection2D, PoseStamped, Detection2D, PoseStamped, float]
        ] = []
        for left_det, left_pose in left:
            best_j = -1
            best_err = float("inf")
            for j, (_rdet, right_pose) in enumerate(right):
                if j in used_right:
                    continue
                err = self._dist_xy(left_pose, right_pose)
                if err < best_err:
                    best_err = err
                    best_j = j
            if best_j < 0 or best_err > tol_m:
                continue
            used_right.add(best_j)
            right_det, right_pose = right[best_j]
            matches.append(
                (left_det, left_pose, right_det, right_pose, best_err)
            )
        return matches

    def _average_poses(self, a: PoseStamped, b: PoseStamped) -> PoseStamped:
        out = PoseStamped()
        out.header = a.header
        out.pose.position.x = 0.5 * (a.pose.position.x + b.pose.position.x)
        out.pose.position.y = 0.5 * (a.pose.position.y + b.pose.position.y)
        out.pose.position.z = 0.5 * (a.pose.position.z + b.pose.position.z)
        out.pose.orientation.w = 1.0
        return out

    def _bbox_size_m(
        self,
        cam: str,
        det: Detection2D,
        pose: PoseStamped,
    ) -> Tuple[float, float]:
        """Apparent bbox width/height in meters from depth in the camera optical frame.

        Uses ``size_m = size_px * Z_optical / f``. Returns (nan, nan) if
        intrinsics or TF depth are unavailable.
        """
        K, optical_frame = self._get_intrinsics(cam, det)
        if K is None or not optical_frame:
            return float("nan"), float("nan")
        fx, fy, _cx, _cy = K
        if fx <= 1e-6 or fy <= 1e-6:
            return float("nan"), float("nan")

        ps = PoseStamped()
        ps.header.frame_id = pose.header.frame_id
        ps.header.stamp = Time().to_msg()
        ps.pose = pose.pose
        try:
            in_opt = self._tf_buffer.transform(
                ps, optical_frame, timeout=Duration(seconds=self._tf_timeout)
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"bbox size TF {pose.header.frame_id}->{optical_frame} failed: {exc}"
            )
            return float("nan"), float("nan")

        z = in_opt.pose.position.z
        if z <= 1e-6:
            return float("nan"), float("nan")
        return (
            float(det.bbox.size_x) * z / fx,
            float(det.bbox.size_y) * z / fy,
        )

    def _triangulate_detections(
        self,
        left_det: Detection2D,
        right_det: Detection2D,
        target_frame: str,
    ) -> Tuple[PoseStamped, float]:
        """Triangulate 3D point from L/R bbox centroids (intrinsics + TF).

        Returns (pose in target_frame, residual_m) where residual is the
        distance between the closest points on the two camera rays.
        """
        o_l, d_l = self._detection_ray_in_frame("left", left_det, target_frame)
        o_r, d_r = self._detection_ray_in_frame("right", right_det, target_frame)
        point, residual = self._closest_point_between_rays(o_l, d_l, o_r, d_r)

        pose = PoseStamped()
        pose.header.frame_id = target_frame
        stamp = left_det.header.stamp
        pose.header.stamp = stamp if (stamp.sec or stamp.nanosec) \
            else self.get_clock().now().to_msg()
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = point[2]
        pose.pose.orientation.w = 1.0
        return pose, residual

    def _detection_ray_in_frame(
        self,
        cam: str,
        det: Detection2D,
        target_frame: str,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Back-project bbox center to a ray (origin, direction) in target_frame."""
        K, optical_frame = self._get_intrinsics(cam, det)
        if K is None:
            raise RuntimeError(
                f"No camera_info for '{cam}' and approx intrinsics disabled"
            )
        fx, fy, cx, cy = K
        u = det.bbox.center.position.x
        v = det.bbox.center.position.y

        origin_opt = PoseStamped()
        origin_opt.header.frame_id = optical_frame
        origin_opt.header.stamp = Time().to_msg()
        origin_opt.pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        tip_opt = PoseStamped()
        tip_opt.header = origin_opt.header
        tip_opt.pose = Pose(
            position=Point(
                x=(u - cx) / fx,
                y=(v - cy) / fy,
                z=1.0,
            ),
            orientation=origin_opt.pose.orientation,
        )

        timeout = Duration(seconds=self._tf_timeout)
        origin_tgt = self._tf_buffer.transform(
            origin_opt, target_frame, timeout=timeout)
        tip_tgt = self._tf_buffer.transform(
            tip_opt, target_frame, timeout=timeout)

        ox = origin_tgt.pose.position.x
        oy = origin_tgt.pose.position.y
        oz = origin_tgt.pose.position.z
        dx = tip_tgt.pose.position.x - ox
        dy = tip_tgt.pose.position.y - oy
        dz = tip_tgt.pose.position.z - oz
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm < 1e-12:
            raise RuntimeError(f"Degenerate ray from '{cam}' pixel ({u:.1f},{v:.1f})")
        return (ox, oy, oz), (dx / norm, dy / norm, dz / norm)

    @staticmethod
    def _closest_point_between_rays(
        o1: Tuple[float, float, float],
        d1: Tuple[float, float, float],
        o2: Tuple[float, float, float],
        d2: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], float]:
        """Midpoint of the shortest segment between two rays; residual = segment length.

        Requires both rays to intersect in front of their cameras (s>0, t>0).
        """
        w0x = o1[0] - o2[0]
        w0y = o1[1] - o2[1]
        w0z = o1[2] - o2[2]
        a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
        b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
        c = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
        d = d1[0] * w0x + d1[1] * w0y + d1[2] * w0z
        e = d2[0] * w0x + d2[1] * w0y + d2[2] * w0z
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            raise RuntimeError("Stereo rays are parallel; cannot triangulate")
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        if s <= 0.0 or t <= 0.0:
            raise RuntimeError(
                f"Triangulation behind a camera (s={s:.3f}, t={t:.3f})"
            )
        p1 = (o1[0] + s * d1[0], o1[1] + s * d1[1], o1[2] + s * d1[2])
        p2 = (o2[0] + t * d2[0], o2[1] + t * d2[1], o2[2] + t * d2[2])
        mid = (
            0.5 * (p1[0] + p2[0]),
            0.5 * (p1[1] + p2[1]),
            0.5 * (p1[2] + p2[2]),
        )
        residual = math.sqrt(
            (p1[0] - p2[0]) ** 2
            + (p1[1] - p2[1]) ** 2
            + (p1[2] - p2[2]) ** 2
        )
        return mid, residual

    def _publish_pose_array(self, poses: List[PoseStamped], frame_id: str) -> None:
        if not poses:
            return
        arr = PoseArray()
        arr.header.frame_id = frame_id
        arr.header.stamp = poses[0].header.stamp
        for p in poses:
            arr.poses.append(p.pose)
        self._poses_pub.publish(arr)

    # --------------------------------------------------------------- detection

    def _pick_detection(
        self,
        arr: Detection2DArray,
        class_filter: str,
        index: int,
    ) -> Optional[Detection2D]:
        dets = list(arr.detections)
        if class_filter:
            dets = [
                d for d in dets
                if (d.id == class_filter)
                or any(
                    r.hypothesis.class_id == class_filter for r in d.results
                )
            ]
        if not dets:
            return None
        if index >= 0:
            return dets[index] if index < len(dets) else None

        def score(d: Detection2D) -> float:
            return max((r.hypothesis.score for r in d.results), default=0.0)

        return max(dets, key=score)

    # -------------------------------------------------------------- intrinsics

    def _get_intrinsics(
        self, cam: str, det: Detection2D
    ) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
        """Return ((fx, fy, cx, cy), optical_frame_id) or (None, '')."""
        info = self._camera_info.get(cam)
        if info is not None and len(info.k) >= 9 and abs(info.k[0]) > 1e-6:
            fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
            frame = info.header.frame_id or f"{cam}_camera_optical_frame"
            return (fx, fy, cx, cy), frame

        if not self._use_approx_k:
            return None, ""

        # Prefer actual image size from camera_info when K is missing/zero.
        w, h = self._approx_w, self._approx_h
        if info is not None and info.width > 0 and info.height > 0:
            w, h = int(info.width), int(info.height)
        fx = (w / 2.0) / math.tan(self._approx_hfov / 2.0)
        fy = fx  # assume square pixels
        cx, cy = w / 2.0, h / 2.0
        frame = det.header.frame_id or f"{cam}_camera_optical_frame"
        self.get_logger().warn(
            f"Using approximate intrinsics for '{cam}' "
            f"(fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}). "
            "Calibrate cameras when you can."
        )
        return (fx, fy, cx, cy), frame

    def _detection_to_ground_pose(
        self,
        cam: str,
        det: Detection2D,
        target_frame: str,
        ground_z: float,
    ) -> PoseStamped:
        K, frame_id = self._get_intrinsics(cam, det)
        if K is None:
            raise RuntimeError(
                f"No camera_info for '{cam}' and approx intrinsics disabled"
            )
        u = det.bbox.center.position.x
        v = det.bbox.center.position.y
        return self._pixel_to_ground_pose(
            u=u, v=v, K=K, optical_frame=frame_id,
            target_frame=target_frame, ground_z=ground_z,
            stamp=det.header.stamp,
        )

    # ---------------------------------------------------------------- project

    def _pixel_to_ground_pose(
        self,
        u: float,
        v: float,
        K: Tuple[float, float, float, float],
        optical_frame: str,
        target_frame: str,
        ground_z: float,
        stamp,
    ) -> PoseStamped:
        fx, fy, cx, cy = K
        # Ray in optical frame: Z forward, X right, Y down (REP-103).
        direction_opt = Vector3(
            x=(u - cx) / fx,
            y=(v - cy) / fy,
            z=1.0,
        )
        origin_opt = PoseStamped()
        origin_opt.header.frame_id = optical_frame
        origin_opt.header.stamp = stamp
        origin_opt.pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

        tip_opt = PoseStamped()
        tip_opt.header = origin_opt.header
        tip_opt.pose = Pose(
            position=Point(
                x=direction_opt.x, y=direction_opt.y, z=direction_opt.z),
            orientation=origin_opt.pose.orientation,
        )

        timeout = Duration(seconds=self._tf_timeout)
        origin_opt.header.stamp = Time().to_msg()  # latest TF
        tip_opt.header.stamp = origin_opt.header.stamp
        origin_tgt = self._tf_buffer.transform(
            origin_opt, target_frame, timeout=timeout)
        tip_tgt = self._tf_buffer.transform(
            tip_opt, target_frame, timeout=timeout)

        ox = origin_tgt.pose.position.x
        oy = origin_tgt.pose.position.y
        oz = origin_tgt.pose.position.z
        dx = tip_tgt.pose.position.x - ox
        dy = tip_tgt.pose.position.y - oy
        dz = tip_tgt.pose.position.z - oz

        if abs(dz) < 1e-9:
            raise RuntimeError(
                "Ray is parallel to the ground plane (dz≈0). "
                "Check camera pitch TF or pixel location."
            )
        t = (ground_z - oz) / dz
        if t <= 0.0:
            raise RuntimeError(
                f"Ground intersection is behind the camera (t={t:.3f}). "
                "Pixel may be above the horizon."
            )

        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.header.stamp = stamp if (stamp.sec or stamp.nanosec) \
            else self.get_clock().now().to_msg()
        pose.pose.position.x = ox + t * dx
        pose.pose.position.y = oy + t * dy
        pose.pose.position.z = ground_z
        pose.pose.orientation.w = 1.0
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = GroundPoseNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
