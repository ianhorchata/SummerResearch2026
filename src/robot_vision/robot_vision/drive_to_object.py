"""Closed-loop drive client: stop -> sense -> navigate -> repeat -> grab.

Only calls ``/vision/detect_poses`` while the Create 3 is stopped (no motion
blur). After each navigate finishes, waits for settle, re-detects, and
re-aims until the remaining approach distance is within tolerance. When at
the grab standoff, runs the arm grab sequence unless ``auto_grab`` is false.

    ros2 run robot_vision drive_to_object
    ros2 run robot_vision drive_to_object --ros-args -p dry_run:=true
    ros2 run robot_vision drive_to_object --ros-args -p auto_grab:=false
    ros2 run robot_vision drive_to_object --ros-args \\
      -p grab_distance_m:=0.381 -p approach_tol_m:=0.05
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Quaternion
from irobot_create_msgs.action import NavigateToPosition
from nav_msgs.msg import Odometry

from robot_arm.grab_sequence import (
    DEFAULT_DROP,
    DEFAULT_GRAB,
    DEFAULT_GRIPPER_CLOSED,
    DEFAULT_GRIPPER_OPEN,
    DEFAULT_MOVE_MS,
    run_grab_sequence,
)
from robot_interfaces.srv import DetectObjectPoses, MoveJoints
from vision_msgs.msg import Detection2D


# 15 inches — default standoff from Create 3 center for grabbing.
_DEFAULT_GRAB_M = 15.0 * 0.0254  # 0.381 m


class DriveToObject(Node):
    def __init__(self):
        super().__init__("drive_to_object")

        self.declare_parameter("stereo", True)
        self.declare_parameter("camera", "left")  # used when stereo=false
        self.declare_parameter("confidence", 0.0)
        self.declare_parameter("match_tol_m", 0.50)
        self.declare_parameter("grab_distance_m", _DEFAULT_GRAB_M)
        self.declare_parameter("max_range_m", 3.0)
        self.declare_parameter("min_range_m", 0.20)
        # Reject objects whose apparent bbox max(width, height) exceeds this.
        self.declare_parameter("max_size_m", 0.20)
        # Done when |object_range - grab| and planar drive remaining are small.
        self.declare_parameter("approach_tol_m", 0.05)
        self.declare_parameter("max_iterations", 8)
        self.declare_parameter("settle_sec", 0.8)
        self.declare_parameter("still_lin_vel_mps", 0.02)
        self.declare_parameter("still_ang_vel_rps", 0.05)
        self.declare_parameter("still_timeout_sec", 15.0)
        self.declare_parameter("max_translation_speed", 0.25)
        self.declare_parameter("max_rotation_speed", 1.0)
        self.declare_parameter("achieve_goal_heading", True)
        self.declare_parameter("dry_run", False)
        # After reaching grab standoff, run arm grab sequence (set false to debug drive).
        self.declare_parameter("auto_grab", True)
        self.declare_parameter("grab_pose", DEFAULT_GRAB)
        self.declare_parameter("drop_pose", DEFAULT_DROP)
        self.declare_parameter("gripper_closed_rad", DEFAULT_GRIPPER_CLOSED)
        self.declare_parameter("gripper_open_rad", DEFAULT_GRIPPER_OPEN)
        self.declare_parameter("arm_move_ms", DEFAULT_MOVE_MS)
        self.declare_parameter("move_joints_service", "/arm/move_joints")
        self.declare_parameter("detect_timeout_sec", 90.0)
        self.declare_parameter("nav_timeout_sec", 120.0)
        self.declare_parameter("odom_topic", "/create3/odom")
        self.declare_parameter(
            "navigate_action", "/create3/navigate_to_position")

        self._stereo = self._as_bool(self.get_parameter("stereo").value)
        self._camera = str(self.get_parameter("camera").value)
        self._confidence = float(self.get_parameter("confidence").value)
        self._match_tol = float(self.get_parameter("match_tol_m").value)
        self._grab_m = float(self.get_parameter("grab_distance_m").value)
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._min_range = float(self.get_parameter("min_range_m").value)
        self._max_size = float(self.get_parameter("max_size_m").value)
        self._approach_tol = float(self.get_parameter("approach_tol_m").value)
        self._max_iters = int(self.get_parameter("max_iterations").value)
        self._settle_sec = float(self.get_parameter("settle_sec").value)
        self._still_lin = float(self.get_parameter("still_lin_vel_mps").value)
        self._still_ang = float(self.get_parameter("still_ang_vel_rps").value)
        self._still_timeout = float(
            self.get_parameter("still_timeout_sec").value)
        self._max_tx = float(self.get_parameter("max_translation_speed").value)
        self._max_rot = float(self.get_parameter("max_rotation_speed").value)
        self._achieve_heading = self._as_bool(
            self.get_parameter("achieve_goal_heading").value)
        self._dry_run = self._as_bool(self.get_parameter("dry_run").value)
        self._auto_grab = self._as_bool(self.get_parameter("auto_grab").value)
        self._grab_pose = [float(x) for x in self.get_parameter("grab_pose").value]
        self._drop_pose = [float(x) for x in self.get_parameter("drop_pose").value]
        self._gripper_closed = float(
            self.get_parameter("gripper_closed_rad").value)
        self._gripper_open = float(self.get_parameter("gripper_open_rad").value)
        self._arm_move_ms = int(self.get_parameter("arm_move_ms").value)
        self._detect_timeout = float(
            self.get_parameter("detect_timeout_sec").value)
        self._nav_timeout = float(self.get_parameter("nav_timeout_sec").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        nav_action = str(self.get_parameter("navigate_action").value)
        move_joints_svc = str(self.get_parameter("move_joints_service").value)

        self._cb_group = ReentrantCallbackGroup()
        self._odom: Optional[Odometry] = None
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Odometry,
            odom_topic,
            self._on_odom,
            sensor_qos,
            callback_group=self._cb_group,
        )

        self._detect_cli = self.create_client(
            DetectObjectPoses,
            "vision/detect_poses",
            callback_group=self._cb_group,
        )
        self._nav_cli = ActionClient(
            self,
            NavigateToPosition,
            nav_action,
            callback_group=self._cb_group,
        )
        self._move_cli = self.create_client(
            MoveJoints,
            move_joints_svc,
            callback_group=self._cb_group,
        )
        self._goal_pub = self.create_publisher(
            PoseStamped, "vision/drive_goal", 10)

        mode = "STEREO" if self._stereo else f"cam={self._camera}"
        self.get_logger().info(
            f"drive_to_object ready ({mode}, grab={self._grab_m:.3f}m / "
            f"{self._grab_m / 0.0254:.1f}in, max_size={self._max_size:.3f}m, "
            f"tol={self._approach_tol:.3f}m, max_iters={self._max_iters}, "
            f"auto_grab={self._auto_grab}, dry_run={self._dry_run})"
        )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _on_odom(self, msg: Odometry):
        self._odom = msg

    # ------------------------------------------------------------------ public

    def run(self) -> int:
        """Closed loop: stop -> sense -> drive until remaining ≈ 0."""
        if not self._wait_for_odom(5.0):
            self.get_logger().error("No /create3/odom yet — is Create 3 up?")
            return 1

        if not self._detect_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/vision/detect_poses not available")
            return 1

        for iteration in range(1, self._max_iters + 1):
            self.get_logger().info(
                f"===== iteration {iteration}/{self._max_iters} ====="
            )

            # Sense only when fully stopped (avoids motion blur).
            if not self._wait_until_still():
                self.get_logger().error("Robot never settled — aborting")
                return 1
            if self._settle_sec > 0.0:
                self.get_logger().info(
                    f"Settled; waiting {self._settle_sec:.1f}s before capture"
                )
                time.sleep(self._settle_sec)

            sense = self._sense_once()
            if sense is None:
                return 1
            det, ox, oy, range_m, remaining_m, yaw, already_close, w_m, h_m = sense

            score = max(
                (r.hypothesis.score for r in det.results), default=float("nan"))
            oid = det.id if det.id else "?"
            range_err = abs(range_m - self._grab_m)
            self.get_logger().info(
                f"Best id={oid} score={score:.2f} "
                f"pose_base_link=(x={ox:.3f}, y={oy:.3f}) "
                f"size=({w_m:.3f}x{h_m:.3f})m  "
                f"range={range_m:.3f}m  range_err={range_err:.3f}m  "
                f"remaining_drive={remaining_m:.3f}m"
            )

            if remaining_m <= self._approach_tol and range_err <= self._approach_tol:
                self.get_logger().info(
                    f"At grab standoff "
                    f"(remaining_drive={remaining_m:.3f}m, "
                    f"|range-grab|={range_err:.3f}m <= {self._approach_tol:.3f}m)"
                )
                return self._finish_with_grab()

            if already_close and remaining_m <= self._approach_tol:
                # Range is at/under grab; no translation left — face object once.
                self.get_logger().info(
                    "Within grab distance; facing object then done"
                )

            approach = self._approach_in_base_link(ox, oy)
            if approach is None:
                return 1
            gx, gy, yaw, _ = approach
            goal_odom = self._base_xy_yaw_to_odom(gx, gy, yaw)
            if goal_odom is None:
                return 1

            self._goal_pub.publish(goal_odom)
            self.get_logger().info(
                f"Approach goal odom=(x={goal_odom.pose.position.x:.3f}, "
                f"y={goal_odom.pose.position.y:.3f}) yaw={yaw:.2f}rad"
            )

            if self._dry_run:
                self.get_logger().info(
                    "dry_run=true — stopping after one sense (no drive)"
                )
                return 0

            # Drive (no cameras / detect during this).
            nav_code = self._navigate(goal_odom)
            if nav_code != 0:
                return nav_code

            # If we only needed a rotate-in-place while already close, stop.
            if already_close:
                self.get_logger().info("Faced object at grab distance — done")
                return self._finish_with_grab()

        self.get_logger().error(
            f"Did not converge within {self._max_iters} iterations"
        )
        return 1

    def _finish_with_grab(self) -> int:
        """After approach success: optionally run the arm grab sequence."""
        if not self._auto_grab:
            self.get_logger().info("auto_grab=false — skipping arm grab")
            return 0
        if self._dry_run:
            self.get_logger().info("dry_run=true — skipping arm grab")
            return 0

        if not self._wait_until_still():
            self.get_logger().error("Robot never settled before grab — aborting")
            return 1
        if self._settle_sec > 0.0:
            self.get_logger().info(
                f"Settled at standoff; waiting {self._settle_sec:.1f}s before grab"
            )
            time.sleep(self._settle_sec)

        if not self._move_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/arm/move_joints not available — cannot grab")
            return 1

        try:
            run_grab_sequence(
                self,
                self._move_cli,
                grab_pose=self._grab_pose,
                drop_pose=self._drop_pose,
                gripper_closed_rad=self._gripper_closed,
                gripper_open_rad=self._gripper_open,
                move_ms=self._arm_move_ms,
                dry_run=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Arm grab failed: {exc}")
            return 1
        return 0

    def _sense_once(
        self,
    ) -> Optional[
        Tuple[Detection2D, float, float, float, float, float, bool, float, float]
    ]:
        """Detect while stopped.

        Returns det, ox, oy, range, remaining, yaw, close, width_m, height_m.
        """
        resp = self._call_detect_poses()
        if resp is None:
            return None
        if not resp.success:
            self.get_logger().error(f"detect_poses failed: {resp.message}")
            return None

        chosen = self._pick_best(
            resp.detections.detections,
            resp.poses,
            list(resp.widths_m),
            list(resp.heights_m),
        )
        if chosen is None:
            self.get_logger().error(
                "No grab-able object "
                f"(need x>0, range in [{self._min_range:.2f}, "
                f"{self._max_range:.2f}] m, "
                f"max(size) <= {self._max_size:.2f} m)"
            )
            return None

        det, pose, w_m, h_m = chosen
        ox = pose.pose.position.x
        oy = pose.pose.position.y
        range_m = math.hypot(ox, oy)
        approach = self._approach_in_base_link(ox, oy)
        if approach is None:
            return None
        gx, gy, yaw, already_close = approach
        remaining_m = math.hypot(gx, gy)
        return det, ox, oy, range_m, remaining_m, yaw, already_close, w_m, h_m

    # ------------------------------------------------------------------ detect

    def _call_detect_poses(self):
        req = DetectObjectPoses.Request()
        req.stereo = self._stereo
        req.camera = self._camera
        req.confidence = self._confidence
        req.target_frame = "base_link"
        req.ground_z = 0.0
        req.match_tol_m = self._match_tol

        self.get_logger().info(
            f"Calling /vision/detect_poses "
            f"(stereo={self._stereo}, match_tol={self._match_tol:.2f}m)..."
        )
        future = self._detect_cli.call_async(req)
        deadline = time.monotonic() + self._detect_timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                self.get_logger().error("detect_poses timed out")
                return None
            time.sleep(0.05)
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"detect_poses exception: {exc}")
            return None

    def _pick_best(
        self,
        detections,
        poses,
        widths_m,
        heights_m,
    ) -> Optional[Tuple[Detection2D, PoseStamped, float, float]]:
        """Highest score among objects in front, in range, and grab-sized."""
        best = None
        best_score = -1.0
        n = min(len(detections), len(poses))
        for i in range(n):
            det = detections[i]
            pose = poses[i]
            w_m = float(widths_m[i]) if i < len(widths_m) else float("nan")
            h_m = float(heights_m[i]) if i < len(heights_m) else float("nan")
            x = pose.pose.position.x
            y = pose.pose.position.y
            r = math.hypot(x, y)
            if x <= 0.0:
                continue
            if r < self._min_range or r > self._max_range:
                continue
            # Reject oversized objects; if size unknown, skip rather than grab.
            if not math.isfinite(w_m) or not math.isfinite(h_m):
                self.get_logger().warn(
                    f"Skipping id={det.id or '?'} — size unknown"
                )
                continue
            if max(w_m, h_m) > self._max_size:
                self.get_logger().info(
                    f"Skipping id={det.id or '?'} "
                    f"size=({w_m:.3f}x{h_m:.3f})m > max_size={self._max_size:.3f}m"
                )
                continue
            score = max(
                (res.hypothesis.score for res in det.results), default=0.0)
            if score > best_score:
                best_score = score
                best = (det, pose, w_m, h_m)
        return best

    # ----------------------------------------------------------- approach math

    def _approach_in_base_link(
        self, ox: float, oy: float
    ) -> Optional[Tuple[float, float, float, bool]]:
        """Return (goal_x, goal_y, yaw, already_close) in base_link."""
        d = math.hypot(ox, oy)
        if d < 1e-6:
            self.get_logger().error("Object is at robot origin — cannot aim")
            return None
        yaw = math.atan2(oy, ox)
        if d <= self._grab_m:
            return 0.0, 0.0, yaw, True
        ux, uy = ox / d, oy / d
        gx = ox - self._grab_m * ux
        gy = oy - self._grab_m * uy
        return gx, gy, yaw, False

    def _wait_for_odom(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self._odom is None:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return self._odom is not None

    def _is_still(self) -> bool:
        if self._odom is None:
            return False
        tw = self._odom.twist.twist
        lin = math.hypot(tw.linear.x, tw.linear.y)
        ang = abs(tw.angular.z)
        return lin <= self._still_lin and ang <= self._still_ang

    def _wait_until_still(self) -> bool:
        """Block until odom twist is near zero (robot fully stopped)."""
        deadline = time.monotonic() + self._still_timeout
        still_since: Optional[float] = None
        need_still_for = 0.25  # require continuous stillness briefly
        while rclpy.ok():
            if self._is_still():
                now = time.monotonic()
                if still_since is None:
                    still_since = now
                elif now - still_since >= need_still_for:
                    return True
            else:
                still_since = None
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return False

    def _base_xy_yaw_to_odom(
        self, x_b: float, y_b: float, yaw_b: float
    ) -> Optional[PoseStamped]:
        """Transform a base_link planar pose into odom using latest odometry."""
        if self._odom is None:
            self.get_logger().error("Lost odom")
            return None
        op = self._odom.pose.pose.position
        oq = self._odom.pose.pose.orientation
        yaw_r = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
        c, s = math.cos(yaw_r), math.sin(yaw_r)
        x_o = op.x + c * x_b - s * y_b
        y_o = op.y + s * x_b + c * y_b
        yaw_o = yaw_r + yaw_b

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "odom"
        goal.pose.position.x = x_o
        goal.pose.position.y = y_o
        goal.pose.position.z = 0.0
        goal.pose.orientation = self._quat_from_yaw(yaw_o)
        return goal

    @staticmethod
    def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _quat_from_yaw(yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    # ---------------------------------------------------------------- navigate

    def _navigate(self, goal_pose: PoseStamped) -> int:
        if not self._nav_cli.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                "NavigateToPosition action server not available "
                f"({self.get_parameter('navigate_action').value})"
            )
            return 1

        goal = NavigateToPosition.Goal()
        goal.goal_pose = goal_pose
        goal.achieve_goal_heading = self._achieve_heading
        goal.max_translation_speed = self._max_tx
        goal.max_rotation_speed = self._max_rot

        self.get_logger().info(
            "Sending NavigateToPosition (no detection while moving)..."
        )
        send_future = self._nav_cli.send_goal_async(
            goal, feedback_callback=self._on_nav_feedback)
        deadline = time.monotonic() + self._nav_timeout
        while rclpy.ok() and not send_future.done():
            if time.monotonic() > deadline:
                self.get_logger().error("Timed out waiting for goal accept")
                return 1
            time.sleep(0.05)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("NavigateToPosition goal rejected")
            return 1

        self.get_logger().info("Goal accepted — driving")
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if time.monotonic() > deadline:
                self.get_logger().error("NavigateToPosition timed out")
                goal_handle.cancel_goal_async()
                return 1
            time.sleep(0.05)

        result = result_future.result()
        if result is None:
            self.get_logger().error("No navigate result")
            return 1

        status = result.status
        # 4 = STATUS_SUCCEEDED in action_msgs/GoalStatus
        if status != 4:
            self.get_logger().error(
                f"NavigateToPosition finished with status={status}"
            )
            return 1

        final = result.result.pose
        self.get_logger().info(
            f"Nav done odom=(x={final.pose.position.x:.3f}, "
            f"y={final.pose.position.y:.3f})"
        )
        return 0

    def _on_nav_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        state = {
            1: "ROTATING_TO_GOAL",
            2: "DRIVING_TO_GOAL",
            3: "ROTATING_TO_HEADING",
        }.get(fb.navigate_state, str(fb.navigate_state))
        self.get_logger().info(
            f"  nav {state}  remaining_dist={fb.remaining_travel_distance:.3f}m  "
            f"remaining_angle={fb.remaining_angle_travel:.3f}rad",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = DriveToObject()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        code = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
