"""2D detections -> ground-plane poses (single or stereo-validated).

Services
--------
``/vision/detect_pose`` (DetectObjectPose)
    Pick one Detection2D and project its bbox contact pixel to the floor.

``/vision/detect_poses`` (DetectObjectPoses)
    Project every detection (numeric ID order). With ``stereo:=true``,
    associate left/right detections then triangulate each match from L/R
    bbox pixels + camera intrinsics (ray midpoint in ``target_frame``).
    Also returns apparent bbox ``widths_m`` / ``heights_m``.

Pipeline (per camera)
---------------------
1. Call ``/vision/detect``.
2. Take bbox bottom-center (or center) and optionally undistort with D.
3. Back-project through camera intrinsics K.
4. Transform the ray into ``target_frame`` (default ``base_link``).
5. Intersect with the horizontal plane z = ground_z.

Stereo association (``stereo:=true``)
-------------------------------------
Default ``match_mode:=epipolar``:
  1. Build fundamental matrix F from stereo CameraInfo R/P when available
     (falls back to K + optical-frame TF for toe-in rigs).
  2. Global min-cost (Hungarian) left/right assignment: epipolar distance +
     nonlinear bbox size + grayscale NCC appearance always in the assign cost.
     Reported ``match_cost`` stays geometric only (epi + size).
  3. Triangulate each pair (bbox bottom = contact height).
  4. Optionally drop poses whose contact_z is off the floor
     (``require_on_ground``, ``ground_contact_z_min/max``).

``match_mode:=ground`` keeps the older L/R ground-XY nearest match within
``match_tol_m``.

    ros2 service call /vision/detect_poses robot_interfaces/srv/DetectObjectPoses \\
      "{stereo: true, match_tol_m: 0.15}"
    ros2 service call /vision/detect_poses robot_interfaces/srv/DetectObjectPoses \\
      "{stereo: false, camera: 'left'}"
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion, Vector3
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401  # registers PoseStamped <-> TF types

from robot_interfaces.srv import DetectObjectPose, DetectObjectPoses, DetectObjects

# Match tuple: left_det, left_pose|None, right_det, right_pose|None,
# err_m, match_cost, debug
StereoMatch = Tuple[
    Detection2D,
    Optional[PoseStamped],
    Detection2D,
    Optional[PoseStamped],
    float,
    float,
    str,
]


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
        # Undistort detection pixels with camera_info.D before back-projection.
        self.declare_parameter("undistort", True)
        # Which bbox pixel hits the floor: "bottom" (contact) or "center".
        self.declare_parameter("project_bbox_point", "bottom")
        # Stereo association: "epipolar" (default) or "ground".
        self.declare_parameter("match_mode", "epipolar")
        # Hand-tuned TF can be ~20-40px off; stereo-cal F is tighter.
        self.declare_parameter("epipolar_tol_px", 40.0)
        # Appearance on image crops (needs */image_raw cache). Always included
        # in Hungarian assign cost; not included in reported match_cost.
        self.declare_parameter("use_appearance", True)
        # Weight on (1 - NCC) in assign cost. Scaled so a full NCC swing
        # (~1.0) can outweigh ~tens of px of epipolar residual when F/contact
        # points are noisy (true matches near tol, false matches near 0).
        self.declare_parameter("appearance_weight", 40.0)
        # Deprecated: appearance always ranks (kept for launch compatibility).
        self.declare_parameter("appearance_tie_px", 8.0)
        # Hard reject if NCC < this. <=0 disables (default: soft assign only).
        self.declare_parameter("min_appearance", -1.0)
        self.declare_parameter("appearance_crop_size", 32)
        # Shrink bbox crop toward center (drops floor/glare border).
        self.declare_parameter("appearance_inset_frac", 0.20)
        # Nonlinear size cost: size_weight * expm1(size_exp * size_pen).
        # Mild for small pen (~bbox jitter); steep past ~0.8; >=1.0 typically
        # exceeds drive_to_object max_match_cost (~45) even with tiny epi.
        self.declare_parameter("size_weight", 15.0)
        self.declare_parameter("size_exp", 1.5)
        # Keep stereo poses whose triangulated bbox-bottom (contact) z is near
        # the ground plane. Rejects elevated logos / tabletop clutter / bad rays.
        self.declare_parameter("require_on_ground", True)
        self.declare_parameter("ground_contact_z_min", -0.05)
        self.declare_parameter("ground_contact_z_max", 0.15)

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
        self._undistort = self._as_bool(self.get_parameter("undistort").value)
        point = str(self.get_parameter("project_bbox_point").value).strip().lower()
        if point not in ("bottom", "center"):
            self.get_logger().warn(
                f"Unknown project_bbox_point='{point}', using 'bottom'"
            )
            point = "bottom"
        self._project_bbox_point = point
        mode = str(self.get_parameter("match_mode").value).strip().lower()
        if mode not in ("epipolar", "ground"):
            self.get_logger().warn(
                f"Unknown match_mode='{mode}', using 'epipolar'"
            )
            mode = "epipolar"
        self._match_mode = mode
        self._epipolar_tol_px = float(self.get_parameter("epipolar_tol_px").value)
        self._use_appearance = self._as_bool(
            self.get_parameter("use_appearance").value)
        self._appearance_weight = float(
            self.get_parameter("appearance_weight").value)
        # appearance_tie_px is deprecated (appearance always ranks); still
        # declared so old launch files do not warn on unknown params.
        self.get_parameter("appearance_tie_px")
        self._min_appearance = float(self.get_parameter("min_appearance").value)
        self._appearance_crop_size = max(
            8, int(self.get_parameter("appearance_crop_size").value))
        self._appearance_inset = float(
            self.get_parameter("appearance_inset_frac").value)
        self._appearance_inset = max(0.0, min(0.45, self._appearance_inset))
        self._size_weight = float(self.get_parameter("size_weight").value)
        self._size_exp = max(0.0, float(self.get_parameter("size_exp").value))
        self._require_on_ground = self._as_bool(
            self.get_parameter("require_on_ground").value)
        self._ground_z_min = float(
            self.get_parameter("ground_contact_z_min").value)
        self._ground_z_max = float(
            self.get_parameter("ground_contact_z_max").value)
        if self._ground_z_min > self._ground_z_max:
            self.get_logger().warn(
                f"ground_contact_z_min ({self._ground_z_min}) > max "
                f"({self._ground_z_max}); swapping"
            )
            self._ground_z_min, self._ground_z_max = (
                self._ground_z_max, self._ground_z_min
            )

        self._cb_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._bridge = CvBridge()
        self._image_lock = threading.Lock()
        self._latest_bgr: dict[str, Optional[np.ndarray]] = {
            "left": None, "right": None,
        }

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
            self.create_subscription(
                Image,
                f"{cam}/image_raw",
                self._make_image_cb(cam),
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
            f"/vision/detect_poses (all / stereo, match_mode={self._match_mode}, "
            f"epi_tol={self._epipolar_tol_px:.1f}px, "
            f"appearance={self._use_appearance}, "
            f"match_tol={self._default_match_tol:.2f}m, "
            f"target_frame={self._default_target}, ground_z={self._default_ground_z}, "
            f"project={self._project_bbox_point}, undistort={self._undistort}, "
            f"on_ground={self._require_on_ground} "
            f"z=[{self._ground_z_min:.2f},{self._ground_z_max:.2f}]m)"
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

    def _make_image_cb(self, cam: str):
        def _cb(msg: Image):
            try:
                bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:  # noqa: BLE001
                return
            with self._image_lock:
                self._latest_bgr[cam] = bgr
        return _cb

    def _cached_bgr(self, cam: str) -> Optional[np.ndarray]:
        with self._image_lock:
            img = self._latest_bgr.get(cam)
            return None if img is None else img

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
        det_req.save_debug = False
        future = self._detect_cli.call_async(det_req)
        return self._await_detect_future(future, cam, timeout_sec)

    def _await_detect_future(self, future, cam: str, timeout_sec: float):
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

    def _call_detect_stereo_pair(
        self, confidence: float, timeout_sec: float = 60.0
    ):
        """Fire left+right ``/vision/detect`` in parallel; wait for both.

        Vision serializes GPU inference; overlapping the second call with the
        first camera's async debug JPEG write still cuts wall time.
        """
        if not self._detect_cli.wait_for_service(timeout_sec=2.0):
            err = "/vision/detect is not available (is vision_node up?)"
            return (False, err, None), (False, err, None)

        req_l = DetectObjects.Request()
        req_l.camera = "left"
        req_l.confidence = confidence
        req_l.save_debug = False
        req_r = DetectObjects.Request()
        req_r.camera = "right"
        req_r.confidence = confidence
        req_r.save_debug = False
        fut_l = self._detect_cli.call_async(req_l)
        fut_r = self._detect_cli.call_async(req_r)
        deadline = time.monotonic() + timeout_sec
        while not (fut_l.done() and fut_r.done()):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        remaining = max(0.5, deadline - time.monotonic())
        left = self._await_detect_future(fut_l, "left", remaining)
        right = self._await_detect_future(fut_r, "right", remaining)
        return left, right

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
        u, v = self._detection_pixel(chosen)
        u_u, v_u = self._undistort_pixel(cam, u, v)
        response.message = (
            f"Projected '{label}' from '{cam}' {self._project_bbox_point} "
            f"pixel ({u:.1f},{v:.1f})"
            + (f" undistorted=({u_u:.1f},{v_u:.1f})" if self._undistort else "")
            + f" -> {target} ({pose.pose.position.x:.3f}, "
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
        response.match_costs = []
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
        (ok_l, msg_l, dets_l), (ok_r, msg_r, dets_r) = self._call_detect_stereo_pair(
            request.confidence
        )
        response.left_detections = dets_l if dets_l is not None else Detection2DArray()
        response.right_detections = dets_r if dets_r is not None else Detection2DArray()
        if not ok_l or dets_l is None:
            response.success = False
            response.message = f"left: {msg_l}"
            return response
        if not ok_r or dets_r is None:
            response.success = False
            response.message = f"right: {msg_r}"
            return response

        left_dets = sorted(list(dets_l.detections), key=self._id_sort_key)
        right_dets = list(dets_r.detections)

        # Ground poses are optional helpers (match_errors_m only; not used as
        # stereo pose substitutes — failed triangulation drops the candidate).
        left_pose_map: dict[int, PoseStamped] = {}
        right_pose_map: dict[int, PoseStamped] = {}
        for det in left_dets:
            try:
                left_pose_map[id(det)] = self._detection_to_ground_pose(
                    "left", det, target, ground_z)
            except Exception:  # noqa: BLE001
                pass
        for det in right_dets:
            try:
                right_pose_map[id(det)] = self._detection_to_ground_pose(
                    "right", det, target, ground_z)
            except Exception:  # noqa: BLE001
                pass

        mode = self._match_mode
        matches: List[StereoMatch] = []
        if mode == "epipolar":
            try:
                matches = self._match_by_epipolar(
                    left_dets, right_dets, left_pose_map, right_pose_map)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"Epipolar match failed ({exc}); falling back to ground XY"
                )
                mode = "ground"

        if mode == "ground":
            left_proj = [
                (d, left_pose_map[id(d)])
                for d in left_dets if id(d) in left_pose_map
            ]
            right_proj = [
                (d, right_pose_map[id(d)])
                for d in right_dets if id(d) in right_pose_map
            ]
            matches = self._match_by_ground_pose(left_proj, right_proj, match_tol)

        out_dets = Detection2DArray()
        out_dets.header = dets_l.header
        poses: List[PoseStamped] = []
        widths: List[float] = []
        heights: List[float] = []
        errors: List[float] = []
        costs: List[float] = []
        kept = 0
        skipped_elevated = 0
        detail_lines: List[str] = []
        for left_det, left_pose, right_det, right_pose, err, cost, match_dbg in matches:
            lid = left_det.id if left_det.id else "?"
            rid = right_det.id if right_det.id else "?"
            contact_z = float("nan")
            height_tri = float("nan")
            try:
                tri_pose, tri_res = self._triangulate_detections(
                    left_det, right_det, target, bbox_point="bottom")
                used = tri_pose
                contact_z = float(tri_pose.pose.position.z)
                height_tri = self._triangulated_height_m(
                    left_det, right_det, target, contact_z)
                range_m = math.hypot(
                    tri_pose.pose.position.x, tri_pose.pose.position.y)
                extra = (
                    f"match=left#{lid}<->right#{rid}  "
                    f"{match_dbg}  "
                    f"match_err={err:.3f}m  "
                    f"tri_residual={tri_res:.3f}m  "
                    f"range_xy={range_m:.3f}m  "
                    f"contact_z={contact_z:.3f}m  "
                    f"height_tri={height_tri:.3f}m"
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"Skip left#{lid}<->right#{rid}: triangulate failed ({exc})"
                )
                continue

            if self._require_on_ground and not self._contact_on_ground(
                contact_z, ground_z
            ):
                skipped_elevated += 1
                self.get_logger().info(
                    f"Skip left#{lid}<->right#{rid}: not on ground "
                    f"(contact_z={contact_z:.3f}m, "
                    f"allow=[{ground_z + self._ground_z_min:.3f}, "
                    f"{ground_z + self._ground_z_max:.3f}]m)"
                )
                continue

            w_m, h_m = self._bbox_size_m("left", left_det, used)
            out_dets.detections.append(left_det)
            poses.append(used)
            widths.append(w_m)
            heights.append(h_m)
            errors.append(float(err))
            costs.append(float(cost))
            kept += 1
            detail_lines.append(
                self._format_object_line(
                    left_det, used, cam="stereo",
                    extra=f"{extra}  size_m=({w_m:.3f}x{h_m:.3f})",
                )
            )

        summary = (
            f"Stereo triangulated {kept}/{len(left_dets)} left object(s) "
            f"(right had {len(right_dets)}; mode={mode}"
            + (
                f", epi_tol={self._epipolar_tol_px:.1f}px"
                if mode == "epipolar"
                else f", tol={match_tol:.2f}m"
            )
            + (
                f", skipped_off_ground={skipped_elevated}"
                if skipped_elevated
                else ""
            )
            + f") -> {target}"
        )
        lines = [summary] + detail_lines

        response.detections = out_dets
        response.poses = poses
        response.widths_m = widths
        response.heights_m = heights
        response.match_errors_m = errors
        response.match_costs = costs
        response.success = True
        response.message = "\n".join(lines)
        for line in lines:
            self.get_logger().info(line)
        self._publish_pose_array(poses, target)
        if poses:
            self._pose_pub.publish(poses[0])
        return response

    def _contact_on_ground(self, contact_z: float, ground_z: float) -> bool:
        """True if triangulated contact height is near the floor plane."""
        if not math.isfinite(contact_z):
            return False
        lo = ground_z + self._ground_z_min
        hi = ground_z + self._ground_z_max
        return lo <= contact_z <= hi

    def _triangulated_height_m(
        self,
        left_det: Detection2D,
        right_det: Detection2D,
        target_frame: str,
        contact_z: float,
    ) -> float:
        """Stereo height ≈ z(top) - z(bottom); nan if top ray fails."""
        try:
            top_pose, _ = self._triangulate_detections(
                left_det, right_det, target_frame, bbox_point="top")
            return float(top_pose.pose.position.z) - float(contact_z)
        except Exception:  # noqa: BLE001
            return float("nan")

    def _format_object_line(
        self,
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
        pu, pv = self._detection_pixel(det)
        px = pose.pose.position.x
        py = pose.pose.position.y
        pz = pose.pose.position.z
        score = max(
            (r.hypothesis.score for r in det.results), default=float("nan"))
        cam_tag = f"cam={cam}  " if cam else ""
        tail = f"  {extra}" if extra else ""
        return (
            f"  {cam_tag}id={oid}  score={score:.2f}  "
            f"center_px=({cx:.1f},{cy:.1f})  "
            f"project_px=({pu:.1f},{pv:.1f})  "
            f"size_px=({sx:.1f}x{sy:.1f})  "
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
    ) -> List[StereoMatch]:
        """Greedy nearest-neighbor match left->right within tol_m (left ID order)."""
        used_right: set[int] = set()
        matches: List[StereoMatch] = []
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
                (
                    left_det, left_pose, right_det, right_pose, best_err,
                    float("nan"),  # no epipolar cost in ground mode
                    f"ground_xy={best_err:.3f}m",
                )
            )
        return matches

    def _match_by_epipolar(
        self,
        left_dets: List[Detection2D],
        right_dets: List[Detection2D],
        left_pose_map: dict,
        right_pose_map: dict,
    ) -> List[StereoMatch]:
        """Global Hungarian match by epi + size + appearance (assign cost)."""
        F, f_src = self._fundamental_matrix()
        img_l = self._cached_bgr("left") if self._use_appearance else None
        img_r = self._cached_bgr("right") if self._use_appearance else None
        if self._use_appearance and (img_l is None or img_r is None):
            self.get_logger().warn(
                "Appearance matching enabled but no cached images; "
                "using epipolar + bbox geometry only"
            )
        self.get_logger().info(
            f"Epipolar Hungarian matching with F from {f_src} "
            f"(tol={self._epipolar_tol_px:.1f}px)"
        )

        n_l = len(left_dets)
        n_r = len(right_dets)
        if n_l == 0 or n_r == 0:
            return []

        # Per-pair features; infeasible slots stay None.
        # (epi, app, size_pen, assign_cost, geo_cost)
        pair: List[List[Optional[Tuple[float, float, float, float, float]]]] = [
            [None] * n_r for _ in range(n_l)
        ]
        # Nearest right (any epi) for unmatched-left diagnostics.
        nearest: List[Tuple[float, str, float, str]] = [
            (float("inf"), "?", float("nan"), "") for _ in range(n_l)
        ]

        for i, left_det in enumerate(left_dets):
            u_l, v_l = self._detection_pixel(left_det)
            u_l, v_l = self._undistort_pixel("left", u_l, v_l)
            for j, right_det in enumerate(right_dets):
                u_r, v_r = self._detection_pixel(right_det)
                u_r, v_r = self._undistort_pixel("right", u_r, v_r)
                epi = self._epipolar_distance_px(F, u_l, v_l, u_r, v_r)
                app = self._appearance_similarity(
                    left_det, right_det, img_l, img_r)
                rid = right_det.id if right_det.id else str(j)
                if epi < nearest[i][0]:
                    nearest[i] = (epi, rid, app, "")

                if epi > self._epipolar_tol_px:
                    if epi <= nearest[i][0] + 1e-9:
                        nearest[i] = (epi, rid, app, "epi>tol")
                    continue

                # Optional hard gate (off by default). Skip weak-appearance
                # candidates that are not uniquely strong on epi.
                if (
                    self._min_appearance > 0.0
                    and self._use_appearance
                    and img_l is not None
                    and img_r is not None
                    and math.isfinite(app)
                    and app < self._min_appearance
                    and epi > 0.5 * self._epipolar_tol_px
                ):
                    if epi <= nearest[i][0] + 1e-9:
                        nearest[i] = (
                            epi,
                            rid,
                            app,
                            f"app={app:.2f}<min={self._min_appearance:.2f}",
                        )
                    continue

                size_pen = self._bbox_size_penalty(left_det, right_det)
                geo = epi + self._size_geo_cost(size_pen)
                # Do not floor negative NCC: anti-correlated crops must cost
                # more than uncorrelated (flooring hid app=-0.15 steal cases).
                if math.isfinite(app):
                    app_clipped = max(-1.0, min(1.0, float(app)))
                else:
                    app_clipped = 0.0
                app_pen = self._appearance_weight * (1.0 - app_clipped)
                assign = geo + app_pen
                pair[i][j] = (epi, app, size_pen, assign, geo)

        # Large finite filler so Hungarian never picks infeasible edges when a
        # feasible alternative exists; unmatched lefts are filtered after.
        big = 1.0e6
        cost = np.full((n_l, n_r), big, dtype=np.float64)
        for i in range(n_l):
            for j in range(n_r):
                cell = pair[i][j]
                if cell is not None:
                    cost[i, j] = cell[3]

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: List[StereoMatch] = []
        assigned_left: set[int] = set()
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            cell = pair[i][j]
            if cell is None or cost[i, j] >= big * 0.5:
                continue
            assigned_left.add(i)
            epi, app, size_pen, assign, geo = cell
            left_det = left_dets[i]
            right_det = right_dets[j]
            left_pose = left_pose_map.get(id(left_det))
            right_pose = right_pose_map.get(id(right_det))
            if left_pose is not None and right_pose is not None:
                err_m = self._dist_xy(left_pose, right_pose)
            else:
                err_m = float("nan")
            # Reported cost is geometric only so drive_to_object's max_match_cost
            # tracks epi quality, not flaky appearance.
            dbg = (
                f"F={f_src}  epi={epi:.1f}px  app={app:.2f}  "
                f"size_pen={size_pen:.2f}  cost={geo:.1f}  "
                f"assign_cost={assign:.1f}"
            )
            matches.append(
                (
                    left_det, left_pose, right_det, right_pose, err_m,
                    float(geo), dbg,
                )
            )

        for i, left_det in enumerate(left_dets):
            if i in assigned_left:
                continue
            lid = left_det.id if left_det.id else "?"
            n_epi, n_rid, n_app, n_why = nearest[i]
            why = f" {n_why}" if n_why else ""
            self.get_logger().warn(
                f"No epipolar match for left#{lid}: nearest right#{n_rid} "
                f"epi={n_epi:.1f}px (tol={self._epipolar_tol_px:.1f}) "
                f"app={n_app:.2f}{why}"
            )
        return matches

    def _fundamental_matrix(self) -> Tuple[np.ndarray, str]:
        """F such that p_R^T F p_L = 0 (pixels, possibly undistorted).

        Prefers stereo CameraInfo R/P (from calibration YAML). Falls back to
        TF that maps left optical -> right optical:
        X_R = R X_L + t, E = [t]_x R, F = K_R^{-T} E K_L^{-1}.
        """
        K_l, frame_l = self._get_intrinsics("left", Detection2D())
        K_r, frame_r = self._get_intrinsics("right", Detection2D())
        if K_l is None or K_r is None:
            raise RuntimeError("Need camera intrinsics for both cameras")

        F_cal = self._fundamental_from_stereo_calib(K_l, K_r)
        if F_cal is not None:
            return F_cal, "stereo_calib"

        if not frame_l:
            frame_l = "left_camera_optical_frame"
        if not frame_r:
            frame_r = "right_camera_optical_frame"

        try:
            tf = self._tf_buffer.lookup_transform(
                frame_r, frame_l, Time(),
                timeout=Duration(seconds=self._tf_timeout),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"TF {frame_l} -> {frame_r} unavailable: {exc}"
            ) from exc

        R = self._quat_to_rot(
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        )
        t = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float64,
        )
        if float(np.linalg.norm(t)) < 1e-9:
            raise RuntimeError("Zero stereo baseline in TF; check camera TFs")

        F = self._essential_to_F(R, t, K_l, K_r)
        return F, "tf"

    def _fundamental_from_stereo_calib(
        self,
        K_l: Tuple[float, float, float, float],
        K_r: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Recover F from CameraInfo rectification R and projection P.

        OpenCV stereoRectify stores R1/R2 and P1/P2 such that the relative
        pose is R = R2^T R1 and t = R2^T * (P2[:,3] / f) (rectified Tx).
        """
        info_l = self._camera_info.get("left")
        info_r = self._camera_info.get("right")
        if info_l is None or info_r is None:
            return None
        if len(info_l.r) < 9 or len(info_r.r) < 9:
            return None
        if len(info_l.p) < 12 or len(info_r.p) < 12:
            return None

        Rl = np.array(info_l.r, dtype=np.float64).reshape(3, 3)
        Rr = np.array(info_r.r, dtype=np.float64).reshape(3, 3)
        Pr = np.array(info_r.p, dtype=np.float64).reshape(3, 4)

        fx = float(Pr[0, 0])
        fy = float(Pr[1, 1])
        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            return None
        # Stereo baseline encoded in P2: typically [fx*Tx, fy*Ty, 0].
        tx = float(Pr[0, 3]) / fx
        ty = float(Pr[1, 3]) / fy
        tz = float(Pr[2, 3])
        t_rect = np.array([tx, ty, tz], dtype=np.float64)
        if float(np.linalg.norm(t_rect)) < 1e-6:
            # No stereo baseline in P — not a stereo calibration.
            return None

        R = Rr.T @ Rl
        t = Rr.T @ t_rect
        if float(np.linalg.norm(t)) < 1e-9:
            return None
        return self._essential_to_F(R, t, K_l, K_r)

    def _essential_to_F(
        self,
        R: np.ndarray,
        t: np.ndarray,
        K_l: Tuple[float, float, float, float],
        K_r: Tuple[float, float, float, float],
    ) -> np.ndarray:
        E = self._skew(t) @ R
        Kl = self._K_matrix(K_l)
        Kr = self._K_matrix(K_r)
        F = np.linalg.inv(Kr).T @ E @ np.linalg.inv(Kl)
        scale = F[2, 2] if abs(F[2, 2]) > 1e-12 else float(np.linalg.norm(F))
        if abs(scale) > 1e-12:
            F = F / scale
        return F

    @staticmethod
    def _K_matrix(K: Tuple[float, float, float, float]) -> np.ndarray:
        fx, fy, cx, cy = K
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    @staticmethod
    def _skew(t: np.ndarray) -> np.ndarray:
        x, y, z = float(t[0]), float(t[1]), float(t[2])
        return np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
        )

    @staticmethod
    def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _epipolar_distance_px(
        F: np.ndarray,
        u_l: float,
        v_l: float,
        u_r: float,
        v_r: float,
    ) -> float:
        """Pixel distance from right point to epipolar line of left point."""
        p_l = np.array([u_l, v_l, 1.0], dtype=np.float64)
        p_r = np.array([u_r, v_r, 1.0], dtype=np.float64)
        line = F @ p_l
        denom = math.hypot(float(line[0]), float(line[1]))
        if denom < 1e-12:
            return float("inf")
        return abs(float(p_r @ line)) / denom

    def _bbox_xyxy(self, det: Detection2D, w: int, h: int) -> Tuple[int, int, int, int]:
        cx = float(det.bbox.center.position.x)
        cy = float(det.bbox.center.position.y)
        sx = float(det.bbox.size_x)
        sy = float(det.bbox.size_y)
        x1 = int(max(0, math.floor(cx - 0.5 * sx)))
        y1 = int(max(0, math.floor(cy - 0.5 * sy)))
        x2 = int(min(w, math.ceil(cx + 0.5 * sx)))
        y2 = int(min(h, math.ceil(cy + 0.5 * sy)))
        if x2 <= x1 or y2 <= y1:
            return 0, 0, 0, 0
        return x1, y1, x2, y2

    def _appearance_similarity(
        self,
        left_det: Detection2D,
        right_det: Detection2D,
        img_l: Optional[np.ndarray],
        img_r: Optional[np.ndarray],
    ) -> float:
        """Grayscale NCC in [-1, 1]; 0 if crops unavailable.

        Mean-normalized cross-correlation on inset bbox crops is far more
        stable than HSV histograms for dark objects / L-R exposure mismatch.
        Class-id bonus applies when labels agree (not generic ``object``).
        """
        sim = 0.0
        if img_l is not None and img_r is not None:
            hl, wl = img_l.shape[:2]
            hr, wr = img_r.shape[:2]
            x1, y1, x2, y2 = self._bbox_xyxy_inset(left_det, wl, hl)
            u1, v1, u2, v2 = self._bbox_xyxy_inset(right_det, wr, hr)
            if x2 > x1 and y2 > y1 and u2 > u1 and v2 > v1:
                crop_l = img_l[y1:y2, x1:x2]
                crop_r = img_r[v1:v2, u1:u2]
                side = self._appearance_crop_size
                gray_l = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)
                gray_l = cv2.resize(
                    gray_l, (side, side), interpolation=cv2.INTER_AREA)
                gray_r = cv2.resize(
                    gray_r, (side, side), interpolation=cv2.INTER_AREA)
                # TM_CCOEFF_NORMED == zero-mean NCC; robust to gain/bias.
                ncc = cv2.matchTemplate(
                    gray_l, gray_r, cv2.TM_CCOEFF_NORMED)
                sim = float(ncc[0, 0]) if ncc.size else 0.0
                if not math.isfinite(sim):
                    sim = 0.0

        cls_l = self._primary_class(left_det)
        cls_r = self._primary_class(right_det)
        if cls_l and cls_r and cls_l == cls_r and cls_l not in ("object", "0"):
            sim = min(1.0, sim + 0.15)
        return sim

    def _bbox_xyxy_inset(
        self, det: Detection2D, w: int, h: int
    ) -> Tuple[int, int, int, int]:
        """BBox crop shrunk toward center by ``appearance_inset_frac``."""
        x1, y1, x2, y2 = self._bbox_xyxy(det, w, h)
        if x2 <= x1 or y2 <= y1:
            return x1, y1, x2, y2
        inset = self._appearance_inset
        if inset <= 0.0:
            return x1, y1, x2, y2
        dx = inset * (x2 - x1)
        dy = inset * (y2 - y1)
        ix1 = int(math.floor(x1 + dx))
        iy1 = int(math.floor(y1 + dy))
        ix2 = int(math.ceil(x2 - dx))
        iy2 = int(math.ceil(y2 - dy))
        if ix2 <= ix1 or iy2 <= iy1:
            return x1, y1, x2, y2
        return ix1, iy1, ix2, iy2

    @staticmethod
    def _primary_class(det: Detection2D) -> str:
        if not det.results:
            return ""
        return str(det.results[0].hypothesis.class_id or "").strip().lower()

    @staticmethod
    def _bbox_size_penalty(a: Detection2D, b: Detection2D) -> float:
        """Soft shape disagreement in ~[0, ~2]; 0 = identical aspect/area ratio."""
        aw = max(float(a.bbox.size_x), 1.0)
        ah = max(float(a.bbox.size_y), 1.0)
        bw = max(float(b.bbox.size_x), 1.0)
        bh = max(float(b.bbox.size_y), 1.0)
        aspect_pen = abs(math.log((aw / ah) / (bw / bh)))
        area_pen = abs(math.log((aw * ah) / (bw * bh)))
        # Area differs a lot under stereo foreshortening; weight aspect more.
        return 0.7 * aspect_pen + 0.3 * min(area_pen, 2.0)

    def _size_geo_cost(self, size_pen: float) -> float:
        """Map size_pen -> geo cost contribution (exponential in pen).

        ``size_weight * expm1(size_exp * size_pen)`` keeps small pens cheap
        (foreshortening / bbox jitter) and heavily punishes pen above ~0.8;
        pen >= 1.0 typically clears max_match_cost even with near-zero epi.
        """
        if self._size_weight <= 0.0 or size_pen <= 0.0:
            return 0.0
        return self._size_weight * math.expm1(self._size_exp * size_pen)

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
        bbox_point: Optional[str] = None,
    ) -> Tuple[PoseStamped, float]:
        """Triangulate 3D point from L/R projection pixels (intrinsics + TF).

        ``bbox_point`` overrides ``project_bbox_point`` ("bottom"/"center"/"top").
        Returns (pose in target_frame, residual_m) where residual is the
        distance between the closest points on the two camera rays.
        """
        o_l, d_l = self._detection_ray_in_frame(
            "left", left_det, target_frame, bbox_point=bbox_point)
        o_r, d_r = self._detection_ray_in_frame(
            "right", right_det, target_frame, bbox_point=bbox_point)
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

    def _detection_pixel(
        self,
        det: Detection2D,
        bbox_point: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Pixel used for ground / stereo rays (bbox bottom/center/top)."""
        u = float(det.bbox.center.position.x)
        v = float(det.bbox.center.position.y)
        point = (bbox_point or self._project_bbox_point).strip().lower()
        if point == "bottom":
            v = v + 0.5 * float(det.bbox.size_y)
        elif point == "top":
            v = v - 0.5 * float(det.bbox.size_y)
        # else: center
        return u, v

    def _undistort_pixel(
        self,
        cam: str,
        u: float,
        v: float,
        K: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, float]:
        """Undistort a distorted pixel into the same K pixel frame.

        Uses ``cv2.undistortPoints(..., P=K)`` so callers can keep using
        ``(u-cx)/fx``. No-ops when undistort is disabled or D is missing.
        """
        if not self._undistort:
            return float(u), float(v)
        info = self._camera_info.get(cam)
        if info is None or len(info.d) < 4:
            return float(u), float(v)
        if not any(abs(float(d)) > 1e-12 for d in info.d):
            return float(u), float(v)
        if K is None:
            if len(info.k) < 9 or abs(info.k[0]) <= 1e-6:
                return float(u), float(v)
            K = (info.k[0], info.k[4], info.k[2], info.k[5])
        fx, fy, cx, cy = K
        Kmat = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        D = np.array(info.d, dtype=np.float64).reshape(-1, 1)
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, Kmat, D, P=Kmat)
        return float(und[0, 0, 0]), float(und[0, 0, 1])

    def _detection_ray_in_frame(
        self,
        cam: str,
        det: Detection2D,
        target_frame: str,
        bbox_point: Optional[str] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Back-project bbox pixel to a ray (origin, direction) in target_frame."""
        K, optical_frame = self._get_intrinsics(cam, det)
        if K is None:
            raise RuntimeError(
                f"No camera_info for '{cam}' and approx intrinsics disabled"
            )
        fx, fy, cx, cy = K
        u, v = self._detection_pixel(det, bbox_point=bbox_point)
        u, v = self._undistort_pixel(cam, u, v, K)

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
        u, v = self._detection_pixel(det)
        u, v = self._undistort_pixel(cam, u, v, K)
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
