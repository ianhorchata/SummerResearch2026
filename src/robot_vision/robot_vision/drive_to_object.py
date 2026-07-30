"""Drive-to-object client with navigate or closed-loop servo control.

``control_mode:=navigate`` (default)
    Stop → sense → Create 3 NavigateToPosition → repeat → align → grab.
    Vision only runs while stopped; motion is open-loop on odometry.
    With ``use_ik:=true``, if the tip is past arm reach at standoff the
    node creeps forward (``ik_creep_step_m``) and re-solves until reachable.
    Each stop takes ``sense_frames`` detects and locks the first chosen object
    in odom so later senses do not hop to a different nearest detection.

``control_mode:=servo``
    Sense → publish ``cmd_vel`` toward the object for ``servo_step_sec`` →
    re-sense → repeat until range + heading are within tolerance, then grab.
    This closes the loop on vision. Stereo FastSAM is slow (~10-20 s/cycle),
    so use ``backend:=blob`` or ``stereo:=false`` for a snappier test.

    ros2 run robot_vision drive_to_object
    ros2 run robot_vision drive_to_object --ros-args -p control_mode:=servo
    ros2 run robot_vision drive_to_object --ros-args -p dry_run:=true
    ros2 run robot_vision drive_to_object --ros-args -p auto_grab:=false
    ros2 run robot_vision drive_to_object --ros-args -p use_ik:=false
    ros2 run robot_vision drive_to_object --ros-args \\
      -p grab_distance_m:=0.381 -p approach_tol_m:=0.05
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
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
from robot_arm.planar_ik import (
    DEFAULT_EE_TIP_XYZ,
    DEFAULT_GRAB_PITCH,
    PlanarArmKinematics,
)
from robot_interfaces.srv import DetectObjectPoses, MoveJoints, PersistDebugImages
from vision_msgs.msg import Detection2D


# 15 inches — default standoff from Create 3 center for grabbing.
_DEFAULT_GRAB_M = 15.0 * 0.0254  # 0.381 m


class DriveToObject(Node):
    def __init__(self, node_name: str = "drive_to_object"):
        super().__init__(node_name)

        self.declare_parameter("stereo", True)
        self.declare_parameter("camera", "left")  # used when stereo=false
        self.declare_parameter("confidence", 0.0)
        self.declare_parameter("match_tol_m", 0.50)
        self.declare_parameter("grab_distance_m", _DEFAULT_GRAB_M)
        self.declare_parameter("max_range_m", 3.0)
        self.declare_parameter("min_range_m", 0.20)
        # Reject objects whose apparent bbox max(width, height) exceeds this.
        self.declare_parameter("max_size_m", 0.15)
        # Reject objects whose apparent bbox min(width, height) is below this.
        # Carpet flecks / baseboard edges often land ~1–2 cm — keep above that.
        self.declare_parameter("min_size_m", 0.01)
        # Reject stereo matches whose geometric cost (epi + size) exceeds this.
        # Keep near epipolar_tol_px (~40) so near-tol matches are not dropped.
        self.declare_parameter("max_match_cost", 35.0)
        # Only pick objects whose stereo contact z is near the floor.
        self.declare_parameter("require_on_ground", True)
        self.declare_parameter("ground_contact_z_min", -0.05)
        self.declare_parameter("ground_contact_z_max", 0.1)
        # Done when |object_range - grab| and planar drive remaining are small.
        self.declare_parameter("approach_tol_m", 0.05)
        # Face object before grab: |atan2(y,x)| must be <= this (rad).
        # ~0.03 rad ≈ 1.1 cm lateral at grab_distance 0.38 m.
        self.declare_parameter("heading_tol_rad", 0.03)
        # Hard lateral gate: planar IK ignores y, so |y| must be this small.
        self.declare_parameter("grab_y_tol_m", 0.012)
        # After a grab, re-sense; if target still near lock, grab again.
        self.declare_parameter("grab_retry_max", 2)
        self.declare_parameter("grab_verify_gate_m", 0.15)
        self.declare_parameter("max_iterations", 8)
        # Extra pause after twist reports still (lets chassis/camera settle).
        # Capture still requires stationary robot; hold+settle reduce blur.
        self.declare_parameter("settle_sec", 0.5)
        self.declare_parameter("still_lin_vel_mps", 0.015)
        self.declare_parameter("still_ang_vel_rps", 0.03)
        # Continuous near-zero twist required before settle_sec starts.
        self.declare_parameter("still_hold_sec", 1.0)
        self.declare_parameter("still_timeout_sec", 20.0)
        self.declare_parameter("max_translation_speed", 0.25)
        self.declare_parameter("max_rotation_speed", 1.0)
        self.declare_parameter("achieve_goal_heading", True)
        # navigate = Create3 NavigateToPosition; servo = cmd_vel + re-sense.
        self.declare_parameter("control_mode", "navigate")
        self.declare_parameter("cmd_vel_topic", "/create3/cmd_vel")
        # How long to drive on each vision update before re-sensing.
        self.declare_parameter("servo_step_sec", 0.8)
        self.declare_parameter("servo_lin_gain", 0.35)
        self.declare_parameter("servo_ang_gain", 1.2)
        # In servo mode, skip long settle so vision updates as often as possible.
        self.declare_parameter("servo_settle_sec", 0.2)
        self.declare_parameter("servo_still_hold_sec", 0.15)
        self.declare_parameter("dry_run", False)
        # After reaching grab standoff, run arm grab sequence (set false to debug drive).
        self.declare_parameter("auto_grab", True)
        # Planar IK from stereo tip (x,z); false keeps taught grab_pose.
        self.declare_parameter("use_ik", True)
        self.declare_parameter("grab_pose", DEFAULT_GRAB)
        self.declare_parameter("drop_pose", DEFAULT_DROP)
        self.declare_parameter("gripper_closed_rad", DEFAULT_GRIPPER_CLOSED)
        self.declare_parameter("gripper_open_rad", DEFAULT_GRIPPER_OPEN)
        self.declare_parameter("arm_move_ms", DEFAULT_MOVE_MS)
        # Wrist pitch (rad); NaN => sum of taught DEFAULT_GRAB arm joints.
        self.declare_parameter("grab_pitch_rad", float("nan"))
        # Fallback tip height if stereo size is unknown (mid-object preferred).
        # Tip is grasp-center; fingers hang below it — keep >= ~4 cm off carpet.
        self.declare_parameter("grab_height_m", 0.04)
        self.declare_parameter("ee_tip_xyz", list(DEFAULT_EE_TIP_XYZ))
        # Deprecated: IK failure always aborts (taught air-grabs disabled).
        self.declare_parameter("ik_fallback_to_taught", False)
        # After standoff, if tip is past arm reach, creep forward and re-try IK.
        self.declare_parameter("ik_creep_step_m", 0.06)
        self.declare_parameter("ik_creep_max_tries", 8)
        self.declare_parameter("move_joints_service", "/arm/move_joints")
        self.declare_parameter("detect_timeout_sec", 90.0)
        self.declare_parameter("nav_timeout_sec", 120.0)
        self.declare_parameter("odom_topic", "/create3/odom")
        self.declare_parameter(
            "navigate_action", "/create3/navigate_to_position")
        # Multi-frame sense while stopped: reduce FastSAM / stereo flicker.
        self.declare_parameter("sense_frames", 3)
        self.declare_parameter("sense_frame_gap_sec", 0.12)
        # Object must appear in >= this many frames (clustered in odom).
        self.declare_parameter("sense_min_hits", 2)
        # Prefer / stick to the first chosen object within this odom gate.
        self.declare_parameter("lock_gate_m", 0.40)
        self.declare_parameter("cluster_tol_m", 0.22)
        # Refuse to hop the lock more than this (furniture FPs nearby).
        self.declare_parameter("lock_jump_max_m", 0.25)
        # After a lock miss, dead-reckon from last lock this many senses.
        self.declare_parameter("lock_miss_max", 2)
        # When matching a locked target, relax size / match_cost filters.
        self.declare_parameter("lock_relax_filters", True)

        self._stereo = self._as_bool(self.get_parameter("stereo").value)
        self._camera = str(self.get_parameter("camera").value)
        self._confidence = float(self.get_parameter("confidence").value)
        self._match_tol = float(self.get_parameter("match_tol_m").value)
        self._grab_m = float(self.get_parameter("grab_distance_m").value)
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._min_range = float(self.get_parameter("min_range_m").value)
        self._max_size = float(self.get_parameter("max_size_m").value)
        self._min_size = float(self.get_parameter("min_size_m").value)
        self._max_match_cost = float(self.get_parameter("max_match_cost").value)
        self._require_on_ground = self._as_bool(
            self.get_parameter("require_on_ground").value)
        self._ground_z_min = float(
            self.get_parameter("ground_contact_z_min").value)
        self._ground_z_max = float(
            self.get_parameter("ground_contact_z_max").value)
        self._approach_tol = float(self.get_parameter("approach_tol_m").value)
        self._heading_tol = float(self.get_parameter("heading_tol_rad").value)
        self._grab_y_tol = max(
            0.005, float(self.get_parameter("grab_y_tol_m").value))
        self._grab_retry_max = max(
            0, int(self.get_parameter("grab_retry_max").value))
        self._grab_verify_gate = max(
            0.05, float(self.get_parameter("grab_verify_gate_m").value))
        self._max_iters = int(self.get_parameter("max_iterations").value)
        self._settle_sec = float(self.get_parameter("settle_sec").value)
        self._still_lin = float(self.get_parameter("still_lin_vel_mps").value)
        self._still_ang = float(self.get_parameter("still_ang_vel_rps").value)
        self._still_hold = float(self.get_parameter("still_hold_sec").value)
        self._still_timeout = float(
            self.get_parameter("still_timeout_sec").value)
        self._max_tx = float(self.get_parameter("max_translation_speed").value)
        self._max_rot = float(self.get_parameter("max_rotation_speed").value)
        self._achieve_heading = self._as_bool(
            self.get_parameter("achieve_goal_heading").value)
        mode = str(self.get_parameter("control_mode").value).strip().lower()
        if mode not in ("navigate", "servo"):
            self.get_logger().warn(
                f"Unknown control_mode '{mode}', using 'navigate'")
            mode = "navigate"
        self._control_mode = mode
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._servo_step = float(self.get_parameter("servo_step_sec").value)
        self._servo_lin_gain = float(self.get_parameter("servo_lin_gain").value)
        self._servo_ang_gain = float(self.get_parameter("servo_ang_gain").value)
        self._servo_settle = float(self.get_parameter("servo_settle_sec").value)
        self._servo_still_hold = float(
            self.get_parameter("servo_still_hold_sec").value)
        self._dry_run = self._as_bool(self.get_parameter("dry_run").value)
        self._auto_grab = self._as_bool(self.get_parameter("auto_grab").value)
        self._use_ik = self._as_bool(self.get_parameter("use_ik").value)
        self._grab_pose = [float(x) for x in self.get_parameter("grab_pose").value]
        self._drop_pose = [float(x) for x in self.get_parameter("drop_pose").value]
        self._gripper_closed = float(
            self.get_parameter("gripper_closed_rad").value)
        self._gripper_open = float(self.get_parameter("gripper_open_rad").value)
        self._arm_move_ms = int(self.get_parameter("arm_move_ms").value)
        pitch_p = float(self.get_parameter("grab_pitch_rad").value)
        self._grab_pitch = (
            DEFAULT_GRAB_PITCH if not math.isfinite(pitch_p) else pitch_p
        )
        self._grab_height = float(self.get_parameter("grab_height_m").value)
        tip = list(self.get_parameter("ee_tip_xyz").value)
        if len(tip) != 3:
            raise ValueError(f"ee_tip_xyz must have 3 values, got {tip}")
        self._ee_tip = (float(tip[0]), float(tip[1]), float(tip[2]))
        self._ik_fallback = self._as_bool(
            self.get_parameter("ik_fallback_to_taught").value)
        self._ik_creep_step = max(
            0.01, float(self.get_parameter("ik_creep_step_m").value))
        self._ik_creep_max = max(
            1, int(self.get_parameter("ik_creep_max_tries").value))
        self._kin = PlanarArmKinematics(ee_tip_xyz=self._ee_tip)
        self._detect_timeout = float(
            self.get_parameter("detect_timeout_sec").value)
        self._nav_timeout = float(self.get_parameter("nav_timeout_sec").value)
        self._sense_frames = max(1, int(self.get_parameter("sense_frames").value))
        self._sense_frame_gap = max(
            0.0, float(self.get_parameter("sense_frame_gap_sec").value))
        self._sense_min_hits = max(
            1, int(self.get_parameter("sense_min_hits").value))
        self._lock_gate = max(0.05, float(self.get_parameter("lock_gate_m").value))
        self._cluster_tol = max(
            0.05, float(self.get_parameter("cluster_tol_m").value))
        self._lock_jump_max = max(
            0.05, float(self.get_parameter("lock_jump_max_m").value))
        self._lock_miss_max = max(
            0, int(self.get_parameter("lock_miss_max").value))
        self._lock_relax = self._as_bool(
            self.get_parameter("lock_relax_filters").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        nav_action = str(self.get_parameter("navigate_action").value)
        move_joints_svc = str(self.get_parameter("move_joints_service").value)

        self._cb_group = ReentrantCallbackGroup()
        self._odom: Optional[Odometry] = None
        # Pose from the last successful scan_for_pickable (base_link).
        # Sweep reuses this for carpet-bounds checks instead of a 2nd detect.
        self._last_scan_pose: Optional[PoseStamped] = None
        self._last_nav_fb = None
        # Locked pick target in odom XY — survives robot motion between senses.
        self._lock_odom_xy: Optional[Tuple[float, float]] = None
        self._lock_size: Optional[Tuple[float, float]] = None
        self._lock_misses = 0
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
        self._persist_debug_cli = self.create_client(
            PersistDebugImages,
            "vision/persist_debug_images",
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
        self._cmd_vel_pub = self.create_publisher(
            Twist, self._cmd_vel_topic, 10)

        cam_mode = "STEREO" if self._stereo else f"cam={self._camera}"
        self.get_logger().info(
            f"drive_to_object ready ({cam_mode}, "
            f"control={self._control_mode}, "
            f"grab={self._grab_m:.3f}m / "
            f"{self._grab_m / 0.0254:.1f}in, "
            f"size=[{self._min_size:.3f},{self._max_size:.3f}]m, "
            f"max_match_cost={self._max_match_cost:.1f}, "
            f"on_ground={self._require_on_ground} "
            f"z=[{self._ground_z_min:.2f},{self._ground_z_max:.2f}]m, "
            f"tol={self._approach_tol:.3f}m, "
            f"heading_tol={self._heading_tol:.3f}rad, "
            f"grab_y_tol={self._grab_y_tol:.3f}m, "
            f"grab_retry={self._grab_retry_max}, "
            f"settle={self._settle_sec:.1f}s, still_hold={self._still_hold:.1f}s, "
            f"sense_frames={self._sense_frames} "
            f"min_hits={self._sense_min_hits} "
            f"lock_gate={self._lock_gate:.2f}m, "
            f"max_iters={self._max_iters}, "
            f"auto_grab={self._auto_grab}, use_ik={self._use_ik}, "
            f"ik_creep={self._ik_creep_step:.3f}m x{self._ik_creep_max}, "
            f"dry_run={self._dry_run})"
        )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _on_odom(self, msg: Odometry):
        self._odom = msg

    # ------------------------------------------------------------------ public

    def pick_one(self) -> int:
        """Approach and grab one visible object. Same as ``run()``.

        Returns 0 on success, 1 on failure, 2 if ``_should_abort_mission``.
        """
        return self.run()

    def scan_for_pickable(self) -> bool:
        """Settle and detect; True if a grab-able object is in view."""
        self._last_scan_pose = None
        self._clear_target_lock()
        if not self._settle_before_sense():
            self.get_logger().warn("scan_for_pickable: never settled")
            return False
        chosen = self._detect_consensus()
        if chosen is None:
            self.get_logger().info("scan_for_pickable: no grab-able object")
            return False
        det, pose, w_m, h_m = chosen
        self._last_scan_pose = pose
        self.get_logger().info(
            f"scan_for_pickable: found id={det.id or '?'} "
            f"at ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}) "
            f"size=({w_m:.3f}x{h_m:.3f})m "
            f"(locked odom={self._lock_odom_xy})"
        )
        return True

    def _should_abort_mission(self) -> bool:
        """Hook for subclasses (battery / servo voltage). Default: never."""
        return False

    def run(self) -> int:
        """Closed loop: stop -> sense -> drive/align until range+heading OK."""
        if not self._wait_for_odom(5.0):
            self.get_logger().error("No /create3/odom yet — is Create 3 up?")
            return 1

        if not self._detect_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/vision/detect_poses not available")
            return 1

        if self._should_abort_mission():
            self.get_logger().warn("Mission abort before pick")
            return 2

        # Keep lock from scan_for_pickable when present; clear when pick ends.
        try:
            if self._control_mode == "servo":
                # Short steps need more vision cycles than navigate mode.
                if self._max_iters < 20:
                    self.get_logger().info(
                        f"servo mode: raising max_iterations "
                        f"{self._max_iters} → 30 (override with -p max_iterations:=N)"
                    )
                    self._max_iters = 30
                return self._run_servo()
            return self._run_navigate()
        finally:
            self._clear_target_lock()

    def _run_navigate(self) -> int:
        """Open-loop Create3 NavigateToPosition between still vision captures."""
        for iteration in range(1, self._max_iters + 1):
            if self._should_abort_mission():
                self.get_logger().warn("Mission abort during navigate pick")
                return 2

            self.get_logger().info(
                f"===== iteration {iteration}/{self._max_iters} ====="
            )

            if not self._settle_before_sense():
                self.get_logger().error("Robot never settled — aborting")
                return 1

            sense = self._sense_once()
            if sense is None:
                return 1
            det, ox, oy, range_m, remaining_m, yaw, already_close, w_m, h_m = sense

            score = max(
                (r.hypothesis.score for r in det.results), default=float("nan"))
            oid = det.id if det.id else "?"
            range_err = abs(range_m - self._grab_m)
            heading_err = abs(math.atan2(oy, ox))
            self.get_logger().info(
                f"Best id={oid} score={score:.2f} "
                f"pose_base_link=(x={ox:.3f}, y={oy:.3f}) "
                f"size=({w_m:.3f}x{h_m:.3f})m  "
                f"range={range_m:.3f}m  range_err={range_err:.3f}m  "
                f"remaining_drive={remaining_m:.3f}m  "
                f"heading_err={heading_err:.3f}rad"
            )

            # Closer than grab standoff is OK (do not require backing up).
            # Old abs(range-grab) check left the robot stuck sending
            # NavigateToPosition to the same XY just to rotate heading.
            at_standoff = range_m <= self._grab_m + self._approach_tol
            aligned = self._grab_aligned(ox, oy)

            if at_standoff and aligned:
                self.get_logger().info(
                    f"At grab standoff and aligned "
                    f"(remaining_drive={remaining_m:.3f}m, "
                    f"|range-grab|={range_err:.3f}m, "
                    f"heading_err={heading_err:.3f}rad, "
                    f"|y|={abs(oy):.3f}m)"
                )
                return self._finish_with_grab()

            if self._dry_run:
                self.get_logger().info(
                    "dry_run=true — stopping after one sense (no drive)"
                )
                return 0

            # At/inside range but facing wrong way: rotate in place, then re-sense.
            if at_standoff and not aligned:
                self.get_logger().info(
                    f"At range but misaligned "
                    f"(heading_err={heading_err:.3f}rad, "
                    f"|y|={abs(oy):.3f}m; "
                    f"need heading<={self._heading_tol:.3f}rad, "
                    f"|y|<={self._grab_y_tol:.3f}m) — rotating to face object"
                )
                nav_code = self._face_object(ox, oy)
                if nav_code != 0:
                    return nav_code
                continue

            approach = self._approach_in_base_link(ox, oy)
            if approach is None:
                return 1
            gx, gy, yaw, already_close = approach
            # Already inside grab distance: never send a zero-translation
            # NavigateToPosition (Create3 often hangs on ROTATING_TO_HEADING).
            if already_close or remaining_m <= self._approach_tol:
                self.get_logger().info(
                    "Inside grab distance — face-object via cmd_vel "
                    "(skip NavigateToPosition)"
                )
                nav_code = self._face_object(ox, oy)
                if nav_code != 0:
                    return nav_code
                continue

            goal_odom = self._base_xy_yaw_to_odom(gx, gy, yaw)
            if goal_odom is None:
                return 1

            self._goal_pub.publish(goal_odom)
            self.get_logger().info(
                f"Approach goal odom=(x={goal_odom.pose.position.x:.3f}, "
                f"y={goal_odom.pose.position.y:.3f}) yaw={yaw:.2f}rad"
            )

            nav_code = self._navigate(goal_odom)
            if nav_code != 0:
                return nav_code
            # Always re-sense after nav (do not grab on already_close alone).

        self.get_logger().error(
            f"Did not converge within {self._max_iters} iterations"
        )
        return 1

    def _run_servo(self) -> int:
        """Vision-closed cmd_vel loop: sense → short drive → re-sense."""
        # Temporarily tighten settle so we re-sense as often as FastSAM allows.
        saved_settle = self._settle_sec
        saved_hold = self._still_hold
        self._settle_sec = max(0.0, self._servo_settle)
        self._still_hold = max(0.05, self._servo_still_hold)
        try:
            for iteration in range(1, self._max_iters + 1):
                if self._should_abort_mission():
                    self.get_logger().warn("Mission abort during servo pick")
                    return 2

                self.get_logger().info(
                    f"===== servo iteration {iteration}/{self._max_iters} ====="
                )

                if not self._settle_before_sense():
                    self.get_logger().error("Robot never settled — aborting")
                    return 1

                sense = self._sense_once()
                if sense is None:
                    self._stop_cmd_vel()
                    return 1
                det, ox, oy, range_m, remaining_m, _yaw, _close, w_m, h_m = sense

                score = max(
                    (r.hypothesis.score for r in det.results),
                    default=float("nan"),
                )
                oid = det.id if det.id else "?"
                range_err = abs(range_m - self._grab_m)
                heading_err = math.atan2(oy, ox)
                self.get_logger().info(
                    f"Best id={oid} score={score:.2f} "
                    f"pose_base_link=(x={ox:.3f}, y={oy:.3f}) "
                    f"size=({w_m:.3f}x{h_m:.3f})m  "
                    f"range={range_m:.3f}m  range_err={range_err:.3f}m  "
                    f"remaining_drive={remaining_m:.3f}m  "
                    f"heading_err={heading_err:.3f}rad"
                )

                # Closer than grab standoff is OK (do not require backing up).
                at_standoff = range_m <= self._grab_m + self._approach_tol
                aligned = self._grab_aligned(ox, oy)

                if at_standoff and aligned:
                    self._stop_cmd_vel()
                    self.get_logger().info(
                        f"At grab standoff and aligned "
                        f"(remaining_drive={remaining_m:.3f}m, "
                        f"|range-grab|={range_err:.3f}m, "
                        f"heading_err={heading_err:.3f}rad, "
                        f"|y|={abs(oy):.3f}m)"
                    )
                    return self._finish_with_grab()

                if self._dry_run:
                    self.get_logger().info(
                        "dry_run=true — stopping after one sense (no drive)"
                    )
                    return 0

                self._servo_step_toward(ox, oy, range_m, heading_err)
            self.get_logger().error(
                f"Did not converge within {self._max_iters} servo iterations"
            )
            return 1
        finally:
            self._stop_cmd_vel()
            self._settle_sec = saved_settle
            self._still_hold = saved_hold

    def _servo_step_toward(
        self, ox: float, oy: float, range_m: float, heading_err: float
    ) -> None:
        """Drive briefly with proportional cmd_vel, then stop for next sense."""
        # Prefer rotating when badly misaligned; otherwise creep toward standoff.
        range_err = range_m - self._grab_m
        ang = max(
            -self._max_rot,
            min(self._max_rot, self._servo_ang_gain * heading_err),
        )
        if abs(heading_err) > 2.0 * self._heading_tol:
            lin = 0.0
        else:
            lin = max(
                -self._max_tx,
                min(self._max_tx, self._servo_lin_gain * range_err),
            )
            # Only drive forward toward the object (never back into it).
            lin = max(0.0, lin)

        twist = Twist()
        twist.linear.x = float(lin)
        twist.angular.z = float(ang)
        self.get_logger().info(
            f"Servo step {self._servo_step:.1f}s: "
            f"lin={lin:.3f}m/s ang={ang:.3f}rad/s "
            f"(heading_err={heading_err:.3f}, range_err={range_err:.3f})"
        )
        deadline = time.monotonic() + max(0.05, self._servo_step)
        rate_period = 0.05
        while rclpy.ok() and time.monotonic() < deadline:
            self._cmd_vel_pub.publish(twist)
            time.sleep(rate_period)
        self._stop_cmd_vel()

    def _stop_cmd_vel(self) -> None:
        self._cmd_vel_pub.publish(Twist())
        # A couple zeros so Create3 definitely sees stop.
        time.sleep(0.05)
        self._cmd_vel_pub.publish(Twist())

    def _finish_with_grab(self) -> int:
        """After approach success: final align, creep until IK-reachable, grab.

        After each grab, re-sense; if the locked target is still nearby, retry
        (planar misses leave the object in place).
        """
        if not self._auto_grab:
            self.get_logger().info("auto_grab=false — skipping arm grab")
            return 0
        if self._dry_run:
            self.get_logger().info("dry_run=true — skipping arm grab")
            return 0

        if not self._move_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/arm/move_joints not available — cannot grab")
            return 1

        max_attempts = 1 + self._grab_retry_max
        last_grab_ok = False
        for attempt in range(1, max_attempts + 1):
            if self._should_abort_mission():
                return 2

            grab_pose = list(self._grab_pose)
            if self._use_ik:
                ik_pose, ik_status = self._resolve_ik_grab_pose()
                if ik_pose is not None:
                    grab_pose = ik_pose
                else:
                    # Never blind-grab with taught pose: "failed" usually means
                    # the lock vanished (furniture/dock FP). Taught grab just
                    # closes on empty air and post-verify falsely says OK.
                    self.get_logger().error(
                        "IK grab failed "
                        f"(status={ik_status}) — aborting pick "
                        "(no taught fallback)"
                    )
                    return 1

            try:
                self.get_logger().info(
                    f"Grab attempt {attempt}/{max_attempts}"
                )
                run_grab_sequence(
                    self,
                    self._move_cli,
                    grab_pose=grab_pose,
                    drop_pose=self._drop_pose,
                    gripper_closed_rad=self._gripper_closed,
                    gripper_open_rad=self._gripper_open,
                    move_ms=self._arm_move_ms,
                    dry_run=False,
                )
                last_grab_ok = True
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Arm grab failed: {exc}")
                return 1

            if attempt >= max_attempts:
                break
            if not self._object_still_at_lock():
                self.get_logger().info(
                    "Post-grab verify: target gone — pick OK"
                )
                return 0
            self.get_logger().warn(
                "Post-grab verify: object still at lock — "
                f"retrying grab ({attempt}/{max_attempts})"
            )

        if last_grab_ok and self._object_still_at_lock():
            self.get_logger().error(
                f"Object still present after {max_attempts} grab attempt(s)"
            )
            return 1
        return 0 if last_grab_ok else 1

    def _object_still_at_lock(self) -> bool:
        """Settle + sense; True if locked target is still visible nearby.

        Disables dead-reckon so an empty scene is not treated as 'still there'.
        """
        if self._lock_odom_xy is None:
            self.get_logger().info(
                "Post-grab verify: no lock — treating as cleared"
            )
            return False
        saved = self._lock_odom_xy
        saved_size = self._lock_size
        if not self._settle_before_sense():
            self.get_logger().warn(
                "Post-grab verify: never settled — treating as cleared"
            )
            return False

        old_miss_max = self._lock_miss_max
        self._lock_miss_max = 0
        try:
            chosen = self._detect_consensus()
        finally:
            self._lock_miss_max = old_miss_max

        if chosen is None:
            self._lock_odom_xy = saved
            self._lock_size = saved_size
            self._lock_misses = 0
            self.get_logger().info(
                "Post-grab verify: no detection at lock — cleared"
            )
            return False

        assert self._lock_odom_xy is not None
        d = math.hypot(
            self._lock_odom_xy[0] - saved[0],
            self._lock_odom_xy[1] - saved[1],
        )
        still = d <= self._grab_verify_gate
        self.get_logger().info(
            f"Post-grab verify: object "
            f"{'still present' if still else 'moved away'} "
            f"d={d:.3f}m gate={self._grab_verify_gate:.2f}m "
            f"odom=({self._lock_odom_xy[0]:.3f}, {self._lock_odom_xy[1]:.3f})"
        )
        if not still:
            self._lock_odom_xy = saved
            self._lock_size = saved_size
        return still

    def _grab_aligned(self, ox: float, oy: float) -> bool:
        """True when heading and lateral |y| are tight enough for planar IK."""
        heading_err = abs(math.atan2(oy, ox))
        return (
            heading_err <= self._heading_tol
            and abs(oy) <= self._grab_y_tol
        )

    def _resolve_ik_grab_pose(self) -> Tuple[Optional[list], str]:
        """Align, solve IK; if tip is past reach, creep forward and retry.

        Returns ``(joints_or_none, status)`` with status ``ok``,
        ``unreachable``, or ``failed``.
        """
        last_status = "failed"
        for attempt in range(1, self._ik_creep_max + 1):
            if not self._align_before_grab():
                return None, "failed"
            if not self._settle_before_sense():
                self.get_logger().error(
                    "Robot never settled before IK capture — aborting"
                )
                return None, "failed"

            ik_pose, status, tip_x = self._solve_ik_grab_pose()
            last_status = status
            if ik_pose is not None:
                return ik_pose, "ok"

            if status == "misaligned":
                # |y| too large for planar IK — re-align without creeping.
                self.get_logger().info(
                    "IK lateral misaligned — re-facing (no creep)"
                )
                continue

            if status != "unreachable":
                # Detect fail / joint limits / etc. — do not creep blindly.
                return None, status

            # Tip beyond arm reach: nudge closer and re-sense.
            if tip_x is not None and tip_x - self._ik_creep_step < self._min_range:
                self.get_logger().error(
                    f"IK unreachable at tip_x={tip_x:.3f}m but creep would "
                    f"pass min_range={self._min_range:.3f}m — aborting"
                )
                return None, "unreachable"
            if attempt >= self._ik_creep_max:
                self.get_logger().error(
                    f"IK still unreachable after {self._ik_creep_max} creep "
                    f"tries (last tip_x={tip_x})"
                )
                return None, "unreachable"

            self.get_logger().info(
                f"IK tip past reach (tip_x={tip_x}); "
                f"creep {self._ik_creep_step:.3f}m forward "
                f"({attempt}/{self._ik_creep_max})"
            )
            if self._creep_forward(self._ik_creep_step) != 0:
                return None, "failed"
        return None, last_status

    def _creep_forward(self, distance_m: float) -> int:
        """Drive straight forward ``distance_m``, actively holding start yaw.

        Uses ``cmd_vel`` (not NavigateToPosition) so Create 3 does not turn
        toward a goal pose mid-creep.
        """
        distance_m = max(0.01, float(distance_m))
        if self._odom is None:
            self.get_logger().error("No odom for IK creep")
            return 1

        op0 = self._odom.pose.pose.position
        oq0 = self._odom.pose.pose.orientation
        x0, y0 = float(op0.x), float(op0.y)
        yaw0 = self._yaw_from_quat(oq0.x, oq0.y, oq0.z, oq0.w)
        speed = min(0.10, self._max_tx)
        # Generous timeout: distance/speed plus settle slack.
        deadline = time.monotonic() + max(2.0, distance_m / max(speed, 1e-3) + 2.0)
        hold_gain = 2.0

        self.get_logger().info(
            f"Creeping forward {distance_m:.3f}m "
            f"(hold yaw={yaw0:.3f}rad)"
        )
        while rclpy.ok() and time.monotonic() < deadline:
            if self._odom is None:
                break
            op = self._odom.pose.pose.position
            oq = self._odom.pose.pose.orientation
            traveled = math.hypot(float(op.x) - x0, float(op.y) - y0)
            if traveled >= distance_m:
                break
            yaw = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
            heading_err = math.atan2(
                math.sin(yaw0 - yaw), math.cos(yaw0 - yaw)
            )
            ang = max(
                -self._max_rot,
                min(self._max_rot, hold_gain * heading_err),
            )
            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = float(ang)
            self._cmd_vel_pub.publish(twist)
            time.sleep(0.05)

        self._stop_cmd_vel()
        return 0

    def _align_before_grab(self, max_tries: int = 5) -> bool:
        """Rotate until heading and |y| are within grab tolerances.

        Aborts if the signed heading keeps flipping — that means the lock
        hopped between nearby furniture false-positives.
        """
        last_yaw_sign = 0
        flips = 0
        for attempt in range(1, max_tries + 1):
            if not self._settle_before_sense():
                self.get_logger().error(
                    "Robot never settled before final align — aborting"
                )
                return False
            sense = self._sense_once()
            if sense is None:
                self.get_logger().error("Final align sense failed")
                return False
            _det, ox, oy, _r, _rem, _yaw, _close, _w, _h = sense
            signed_yaw = math.atan2(oy, ox)
            heading_err = abs(signed_yaw)
            if self._grab_aligned(ox, oy):
                self.get_logger().info(
                    f"Aligned for grab "
                    f"(heading_err={heading_err:.3f}rad, "
                    f"|y|={abs(oy):.3f}m, attempt={attempt})"
                )
                return True

            # Large opposite-direction corrections ⇒ lock is hopping.
            if heading_err > max(0.08, 2.0 * self._heading_tol):
                sign = 1 if signed_yaw > 0.0 else -1
                if last_yaw_sign != 0 and sign != last_yaw_sign:
                    flips += 1
                    self.get_logger().warn(
                        f"Final align yaw flipped "
                        f"({flips}/2, yaw={signed_yaw:.3f}rad)"
                    )
                    if flips >= 2:
                        self.get_logger().error(
                            "Final align oscillating — lock hopping "
                            "between nearby detections; aborting pick"
                        )
                        return False
                last_yaw_sign = sign

            self.get_logger().info(
                f"Final align attempt {attempt}/{max_tries}: "
                f"heading_err={heading_err:.3f}rad |y|={abs(oy):.3f}m "
                f"(need heading<={self._heading_tol:.3f}, "
                f"|y|<={self._grab_y_tol:.3f}) — rotating"
            )
            if self._face_object(ox, oy) != 0:
                return False
        self.get_logger().error(
            f"Could not align within {max_tries} tries "
            f"(need heading_err <= {self._heading_tol:.3f}rad and "
            f"|y| <= {self._grab_y_tol:.3f}m)"
        )
        return False

    def _face_object(self, ox: float, oy: float) -> int:
        """Rotate in place so +X faces the object (no translation).

        Always uses ``cmd_vel``. Create3 ``NavigateToPosition`` often hangs
        forever on ``ROTATING_TO_HEADING`` for small remaining angles when the
        XY goal is already reached.
        """
        yaw = math.atan2(oy, ox)
        self.get_logger().info(
            f"Face-object yaw={yaw:.3f}rad "
            f"(object at x={ox:.3f}, y={oy:.3f})"
        )
        return self._rotate_in_place(yaw)

    def _rotate_in_place(self, delta_yaw: float) -> int:
        """Rotate about Z by ``delta_yaw`` using cmd_vel (odom-closed when available)."""
        if abs(delta_yaw) < 1e-3:
            return 0

        speed = min(self._max_rot, max(0.15, abs(delta_yaw) * 1.2))
        # Open-loop duration cap; odom feedback exits early when close enough.
        duration = min(4.0, abs(delta_yaw) / max(speed, 1e-3) + 0.35)
        yaw0 = None
        if self._odom is not None:
            oq = self._odom.pose.pose.orientation
            yaw0 = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
            target = yaw0 + delta_yaw

        twist = Twist()
        twist.angular.z = math.copysign(speed, delta_yaw)
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            if yaw0 is not None and self._odom is not None:
                oq = self._odom.pose.pose.orientation
                yaw = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
                err = math.atan2(math.sin(target - yaw), math.cos(target - yaw))
                if abs(err) <= self._heading_tol:
                    break
                twist.angular.z = math.copysign(
                    min(speed, max(0.12, abs(err) * 1.5)), err
                )
            self._cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        self._stop_cmd_vel()
        return 0

    def _settle_before_sense(self) -> bool:
        """Wait until still, then pause so capture is not motion-blurred."""
        deadline = time.monotonic() + self._still_timeout
        while rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            if not self._wait_until_still():
                return False
            if self._settle_sec <= 0.0:
                return True
            self.get_logger().info(
                f"Still for {self._still_hold:.1f}s; "
                f"waiting {self._settle_sec:.1f}s before capture"
            )
            time.sleep(self._settle_sec)
            if self._is_still():
                return True
            self.get_logger().warn(
                "Motion detected during settle pause — waiting again"
            )
        return False

    def _solve_ik_grab_pose(
        self,
    ) -> Tuple[Optional[list], str, Optional[float]]:
        """Fresh stereo pose -> planar IK joints.

        Returns ``(joints_or_none, status, tip_x)`` where status is
        ``ok``, ``unreachable`` (tip past arm reach — creep closer), or
        ``failed`` (no detection / joint limits / other).
        """
        # Require a live stereo hit — do not IK on dead-reckoned lock.
        old_miss_max = self._lock_miss_max
        self._lock_miss_max = 0
        try:
            chosen = self._detect_consensus()
        finally:
            self._lock_miss_max = old_miss_max
        if chosen is None:
            self.get_logger().warn("No grab-able object for IK at standoff")
            return None, "failed", None

        det, pose, _w_m, h_m = chosen
        tip_x = float(pose.pose.position.x)
        tip_y = float(pose.pose.position.y)
        contact_z = float(pose.pose.position.z)
        tip_z = self._grab_tip_z(contact_z, h_m)
        if abs(tip_z - contact_z) > 1e-3:
            self.get_logger().info(
                f"Grab tip z={tip_z:.3f}m "
                f"(contact_z={contact_z:.3f}m, height={h_m:.3f}m)"
            )

        # Planar IK ignores y — refuse to grab if still off the arm plane.
        if abs(tip_y) > self._grab_y_tol:
            self.get_logger().warn(
                f"IK skip: |y|={abs(tip_y):.3f}m > "
                f"grab_y_tol={self._grab_y_tol:.3f}m — need re-face"
            )
            if self._face_object(tip_x, tip_y) != 0:
                return None, "failed", tip_x
            return None, "misaligned", tip_x

        try:
            sol = self._kin.ik(
                tip_x,
                tip_z,
                pitch=self._grab_pitch,
                seed=self._grab_pose[:3],
                y=tip_y,
            )
        except ValueError as exc:
            msg = str(exc)
            self.get_logger().warn(f"IK failed: {exc}")
            if "unreachable" in msg.lower():
                return None, "unreachable", tip_x
            return None, "failed", tip_x

        self.get_logger().info(
            f"IK grab tip=({tip_x:.3f}, y={tip_y:.3f}, z={tip_z:.3f}) "
            f"pitch={self._grab_pitch:.3f} -> "
            f"q=[{sol.joints[0]:.3f}, {sol.joints[1]:.3f}, {sol.joints[2]:.3f}] "
            f"(id={det.id or '?'})"
        )
        if abs(tip_y) > 0.05:
            self.get_logger().warn(
                f"|y|={abs(tip_y):.3f}m off arm plane — base heading may be poor"
            )
        return [
            sol.joints[0],
            sol.joints[1],
            sol.joints[2],
            self._gripper_open,
        ], "ok", tip_x

    def _grab_tip_z(self, contact_z: float, height_m: float) -> float:
        """Grasp-center height in base_link.

        Stereo pose ``z`` is the bbox contact (bottom). Prefer mid-object when
        height is known, but never drop the tip below finger clearance — the
        EE tip is the jaw center and fingertips hang ~3–4 cm below it, so a
        mid-object aim of ~2 cm on a short toy drives fingers into the carpet.
        """
        z_floor = max(0.04, float(self._grab_height))
        if math.isfinite(height_m) and height_m > 0.01:
            mid = float(contact_z) + 0.5 * float(height_m)
            return max(mid, z_floor)
        if math.isfinite(contact_z) and contact_z >= z_floor:
            return float(contact_z)
        return z_floor

    def _sense_once(
        self,
    ) -> Optional[
        Tuple[Detection2D, float, float, float, float, float, bool, float, float]
    ]:
        """Detect while stopped (multi-frame + target lock).

        Returns det, ox, oy, range, remaining, yaw, close, width_m, height_m.
        """
        chosen = self._detect_consensus()
        if chosen is None:
            self.get_logger().error(
                "No grab-able object "
                f"(need x>0, range in [{self._min_range:.2f}, "
                f"{self._max_range:.2f}] m, "
                f"max(size) <= {self._max_size:.2f} m, "
                f"min(size) >= {self._min_size:.3f} m, "
                f"match_cost <= {self._max_match_cost:.1f}"
                + (
                    f", lock_gate={self._lock_gate:.2f}m"
                    if self._lock_odom_xy is not None
                    else ""
                )
                + ")"
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

    def _clear_target_lock(self) -> None:
        self._lock_odom_xy = None
        self._lock_size = None
        self._lock_misses = 0

    def _set_target_lock(
        self, ox_b: float, oy_b: float, w_m: float, h_m: float
    ) -> None:
        odom_xy = self._base_xy_to_odom(ox_b, oy_b)
        if odom_xy is None:
            return
        self._lock_odom_xy = odom_xy
        if math.isfinite(w_m) and math.isfinite(h_m):
            self._lock_size = (float(w_m), float(h_m))
        self._lock_misses = 0
        self.get_logger().info(
            f"Target lock odom=({odom_xy[0]:.3f}, {odom_xy[1]:.3f}) "
            f"base=({ox_b:.3f}, {oy_b:.3f}) "
            f"size=({w_m:.3f}x{h_m:.3f})m"
        )

    def _base_xy_to_odom(
        self, x_b: float, y_b: float
    ) -> Optional[Tuple[float, float]]:
        if self._odom is None:
            return None
        op = self._odom.pose.pose.position
        oq = self._odom.pose.pose.orientation
        yaw_r = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
        c, s = math.cos(yaw_r), math.sin(yaw_r)
        return (
            float(op.x) + c * x_b - s * y_b,
            float(op.y) + s * x_b + c * y_b,
        )

    def _odom_xy_to_base(
        self, x_o: float, y_o: float
    ) -> Optional[Tuple[float, float]]:
        if self._odom is None:
            return None
        op = self._odom.pose.pose.position
        oq = self._odom.pose.pose.orientation
        yaw_r = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
        c, s = math.cos(yaw_r), math.sin(yaw_r)
        dx = float(x_o) - float(op.x)
        dy = float(y_o) - float(op.y)
        return (c * dx + s * dy, -s * dx + c * dy)

    @staticmethod
    def _median(vals: List[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return float("nan")
        mid = n // 2
        if n % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    def _valid_candidates(
        self,
        detections,
        poses,
        widths_m,
        heights_m,
        match_costs,
        *,
        relax: bool = False,
    ) -> List[dict]:
        """Filter detections into grab-candidate dicts (base_link + odom)."""
        costs = list(match_costs) if match_costs is not None else []
        max_size = self._max_size * (1.5 if relax and self._lock_relax else 1.0)
        min_size = self._min_size * (0.5 if relax and self._lock_relax else 1.0)
        max_cost = self._max_match_cost * (
            1.5 if relax and self._lock_relax else 1.0
        )
        out: List[dict] = []
        n = min(len(detections), len(poses))
        for i in range(n):
            det = detections[i]
            pose = poses[i]
            w_m = float(widths_m[i]) if i < len(widths_m) else float("nan")
            h_m = float(heights_m[i]) if i < len(heights_m) else float("nan")
            cost = float(costs[i]) if i < len(costs) else float("nan")
            x = float(pose.pose.position.x)
            y = float(pose.pose.position.y)
            r = math.hypot(x, y)
            if x <= 0.0:
                continue
            if r < self._min_range or r > self._max_range:
                continue
            if not math.isfinite(w_m) or not math.isfinite(h_m):
                continue
            if max(w_m, h_m) > max_size:
                continue
            if min(w_m, h_m) < min_size:
                continue
            if math.isfinite(cost) and cost > max_cost:
                continue
            z = float(pose.pose.position.z)
            if self._require_on_ground and not (
                self._ground_z_min <= z <= self._ground_z_max
            ):
                continue
            odom_xy = self._base_xy_to_odom(x, y)
            if odom_xy is None:
                continue
            out.append(
                {
                    "det": det,
                    "pose": pose,
                    "w": w_m,
                    "h": h_m,
                    "cost": cost,
                    "x": x,
                    "y": y,
                    "r": r,
                    "ox": odom_xy[0],
                    "oy": odom_xy[1],
                }
            )
        return out

    def _detect_consensus(
        self,
    ) -> Optional[Tuple[Detection2D, PoseStamped, float, float]]:
        """Multi-frame detect + stick to locked odom target when set.

        While stationary, take ``sense_frames`` stereo detects, require the
        same object (clustered in odom) in ``sense_min_hits`` frames, and
        never hop to a different nearest object once locked.
        """
        n_frames = self._sense_frames
        min_hits = min(n_frames, self._sense_min_hits)
        locked = self._lock_odom_xy is not None
        frames: List[List[dict]] = []
        for i in range(n_frames):
            if i > 0 and self._sense_frame_gap > 0.0:
                time.sleep(self._sense_frame_gap)
            resp = self._call_detect_poses()
            if resp is None or not resp.success:
                self.get_logger().info(
                    f"sense frame {i + 1}/{n_frames}: "
                    f"{getattr(resp, 'message', 'no response')}"
                )
                frames.append([])
                continue
            cands = self._valid_candidates(
                resp.detections.detections,
                resp.poses,
                list(resp.widths_m),
                list(resp.heights_m),
                list(getattr(resp, "match_costs", []) or []),
                relax=locked,
            )
            frames.append(cands)
            self.get_logger().info(
                f"sense frame {i + 1}/{n_frames}: {len(cands)} candidate(s)"
            )

        if locked:
            chosen = self._consensus_locked(frames, min_hits)
        else:
            chosen = self._consensus_unlocked(frames, min_hits)

        if chosen is None:
            if locked and self._lock_misses < self._lock_miss_max:
                self._lock_misses += 1
                self.get_logger().warn(
                    f"Lock miss {self._lock_misses}/{self._lock_miss_max} "
                    "— dead-reckoning last lock"
                )
                return self._sense_from_lock()
            return None

        self._lock_misses = 0
        det = chosen["det"]
        pose = chosen["pose"]
        w_m = float(chosen["w"])
        h_m = float(chosen["h"])
        # Median odom across matched hits → refresh lock + base pose XY/Z.
        hits = chosen.get("hits") or [chosen]
        ox = self._median([float(h["ox"]) for h in hits])
        oy = self._median([float(h["oy"]) for h in hits])
        was_locked = self._lock_odom_xy is not None
        if was_locked:
            assert self._lock_odom_xy is not None
            jump = math.hypot(
                ox - self._lock_odom_xy[0], oy - self._lock_odom_xy[1]
            )
            if jump > self._lock_jump_max:
                # Nearby furniture/baseboard FPs often sit just inside a wide
                # gate; hopping them causes endless left/right face loops.
                self.get_logger().warn(
                    f"Lock jump {jump:.3f}m > "
                    f"{self._lock_jump_max:.2f}m "
                    f"to odom=({ox:.3f}, {oy:.3f}) — refusing hop"
                )
                if self._lock_misses < self._lock_miss_max:
                    self._lock_misses += 1
                    self.get_logger().warn(
                        f"Lock miss {self._lock_misses}/{self._lock_miss_max} "
                        "— dead-reckoning last lock"
                    )
                    return self._sense_from_lock()
                return None
        self._lock_odom_xy = (ox, oy)
        if math.isfinite(w_m) and math.isfinite(h_m):
            self._lock_size = (w_m, h_m)
        base = self._odom_xy_to_base(ox, oy)
        if base is not None:
            pose = PoseStamped()
            pose.header = chosen["pose"].header
            pose.pose.orientation = chosen["pose"].pose.orientation
            pose.pose.position.x = base[0]
            pose.pose.position.y = base[1]
            pose.pose.position.z = self._median(
                [float(h["pose"].pose.position.z) for h in hits]
            )
        self.get_logger().info(
            f"{'Updated' if was_locked else 'Locked'} target "
            f"odom=({ox:.3f}, {oy:.3f}) "
            f"hits={len(hits)}/{n_frames} "
            f"size=({w_m:.3f}x{h_m:.3f})m"
        )
        self._persist_used_debug_images()
        return det, pose, w_m, h_m

    def _persist_used_debug_images(self) -> None:
        """Write buffered annotated debug JPEGs for the sense we just used."""
        if not self._persist_debug_cli.service_is_ready():
            if not self._persist_debug_cli.wait_for_service(timeout_sec=0.5):
                return
        req = PersistDebugImages.Request()
        req.cameras = []  # all buffered cameras (left/right from last hits)
        try:
            future = self._persist_debug_cli.call_async(req)
            deadline = time.monotonic() + 5.0
            while not future.done():
                if time.monotonic() >= deadline:
                    self.get_logger().warn("persist_debug_images timed out")
                    return
                time.sleep(0.02)
            res = future.result()
            if res is not None and res.success:
                self.get_logger().info(
                    f"Saved used debug images: {list(res.saved)}"
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"persist_debug_images failed: {exc}")

    def _sense_from_lock(
        self,
    ) -> Optional[Tuple[Detection2D, PoseStamped, float, float]]:
        """Build a synthetic detection from the odom lock (no vision)."""
        if self._lock_odom_xy is None:
            return None
        base = self._odom_xy_to_base(*self._lock_odom_xy)
        if base is None:
            return None
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = base[0]
        pose.pose.position.y = base[1]
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        det = Detection2D()
        det.id = "lock"
        w_m, h_m = self._lock_size if self._lock_size else (0.05, 0.05)
        return det, pose, w_m, h_m

    def _consensus_locked(
        self, frames: List[List[dict]], min_hits: int
    ) -> Optional[dict]:
        assert self._lock_odom_xy is not None
        lx, ly = self._lock_odom_xy
        # Single gate only — expanding used to latch onto a neighboring
        # furniture FP and left/right face-loop forever.
        gate = self._lock_gate
        hits: List[dict] = []
        for frame in frames:
            best = None
            best_d = float("inf")
            for c in frame:
                d = math.hypot(c["ox"] - lx, c["oy"] - ly)
                if d <= gate and d < best_d:
                    best = c
                    best_d = d
            if best is not None:
                hits.append(best)
        if len(hits) >= min_hits:
            hits_sorted = sorted(
                hits,
                key=lambda c: math.hypot(c["ox"] - lx, c["oy"] - ly),
            )
            rep = dict(hits_sorted[0])
            rep["hits"] = hits
            self.get_logger().info(
                f"Lock match hits={len(hits)}/{len(frames)} "
                f"gate={gate:.2f}m"
            )
            return rep
        self.get_logger().info(
            f"Lock match only {len(hits)}/{min_hits} hits "
            f"inside gate={gate:.2f}m"
        )
        return None

    def _consensus_unlocked(
        self, frames: List[List[dict]], min_hits: int
    ) -> Optional[dict]:
        """Cluster detections across frames in odom; pick nearest stable cluster."""
        # Flatten with frame index so each frame contributes at most once per cluster.
        seeds: List[dict] = []
        for frame in frames:
            seeds.extend(frame)
        if not seeds:
            return None

        used = [False] * len(seeds)
        # Map seed index -> (frame_idx within its frame list) via rebuilding.
        # Simpler: cluster greedily from nearest-to-robot seeds.
        seeds_sorted = sorted(range(len(seeds)), key=lambda i: seeds[i]["r"])
        best_cluster: Optional[List[dict]] = None
        best_key: Optional[Tuple[int, float]] = None  # (-hits, mean_range)

        for si in seeds_sorted:
            if used[si]:
                continue
            seed = seeds[si]
            cluster: List[dict] = []
            # At most one hit per frame.
            for frame in frames:
                best = None
                best_d = float("inf")
                for c in frame:
                    d = math.hypot(c["ox"] - seed["ox"], c["oy"] - seed["oy"])
                    if d <= self._cluster_tol and d < best_d:
                        best = c
                        best_d = d
                if best is not None:
                    cluster.append(best)
            if len(cluster) < min_hits:
                used[si] = True
                continue
            mean_r = sum(c["r"] for c in cluster) / len(cluster)
            key = (-len(cluster), mean_r)
            if best_key is None or key < best_key:
                best_key = key
                best_cluster = cluster
            used[si] = True

        if best_cluster is None:
            # Do not fall back to a single-frame flicker — that starts
            # chases on furniture/dock FPs that vanish at grab range.
            self.get_logger().info(
                "Multi-frame consensus failed — no stable cluster "
                f"(need >={min_hits} hits)"
            )
            return None

        # Representative: median-closest to cluster centroid.
        cx = self._median([c["ox"] for c in best_cluster])
        cy = self._median([c["oy"] for c in best_cluster])
        rep = dict(
            min(
                best_cluster,
                key=lambda c: math.hypot(c["ox"] - cx, c["oy"] - cy),
            )
        )
        rep["hits"] = best_cluster
        self.get_logger().info(
            f"Consensus cluster hits={len(best_cluster)}/{len(frames)} "
            f"odom=({cx:.3f}, {cy:.3f})"
        )
        return rep

    def _pick_best(
        self,
        detections,
        poses,
        widths_m,
        heights_m,
        match_costs=None,
    ) -> Optional[Tuple[Detection2D, PoseStamped, float, float]]:
        """Nearest object among those in front, in range, grab-sized, low cost.

        Prefer locked odom target when set (used by callers that still do
        single-frame picks).
        """
        locked = self._lock_odom_xy is not None
        cands = self._valid_candidates(
            detections,
            poses,
            widths_m,
            heights_m,
            match_costs,
            relax=locked,
        )
        if not cands:
            return None
        if locked:
            lx, ly = self._lock_odom_xy  # type: ignore[misc]
            gated = [
                c
                for c in cands
                if math.hypot(c["ox"] - lx, c["oy"] - ly) <= self._lock_gate
            ]
            if not gated:
                return None
            best = min(
                gated, key=lambda c: math.hypot(c["ox"] - lx, c["oy"] - ly)
            )
        else:
            best = min(cands, key=lambda c: c["r"])
        return best["det"], best["pose"], best["w"], best["h"]

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
        """Block until odom twist stays near zero for ``still_hold_sec``."""
        deadline = time.monotonic() + self._still_timeout
        still_since: Optional[float] = None
        need_still_for = max(0.1, self._still_hold)
        while rclpy.ok():
            if self._is_still():
                now = time.monotonic()
                if still_since is None:
                    still_since = now
                    self.get_logger().info(
                        "Twist near zero — holding for "
                        f"{need_still_for:.1f}s before settle",
                        throttle_duration_sec=2.0,
                    )
                elif now - still_since >= need_still_for:
                    return True
            else:
                if still_since is not None:
                    tw = self._odom.twist.twist if self._odom else None
                    if tw is not None:
                        lin = math.hypot(tw.linear.x, tw.linear.y)
                        ang = abs(tw.angular.z)
                        self.get_logger().info(
                            f"Still moving (lin={lin:.3f}m/s, "
                            f"ang={ang:.3f}rad/s) — waiting",
                            throttle_duration_sec=1.0,
                        )
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
        self._last_nav_fb = None
        send_future = self._nav_cli.send_goal_async(
            goal, feedback_callback=self._on_nav_feedback)
        deadline = time.monotonic() + self._nav_timeout
        while rclpy.ok() and not send_future.done():
            if self._should_abort_mission():
                self.get_logger().warn("Mission abort while waiting for nav accept")
                return 2
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
        # Stall detection: Create3 sometimes spins forever on
        # ROTATING_TO_HEADING with a near-constant remaining_angle.
        last_angle: Optional[float] = None
        stall_since: Optional[float] = None
        while rclpy.ok() and not result_future.done():
            if self._should_abort_mission():
                self.get_logger().warn("Mission abort — canceling NavigateToPosition")
                goal_handle.cancel_goal_async()
                return 2
            if time.monotonic() > deadline:
                self.get_logger().error("NavigateToPosition timed out")
                goal_handle.cancel_goal_async()
                return 1
            fb = self._last_nav_fb
            if fb is not None and float(fb.remaining_travel_distance) < 0.03:
                ang = float(fb.remaining_angle_travel)
                now = time.monotonic()
                if last_angle is not None and abs(ang - last_angle) < 0.02:
                    if stall_since is None:
                        stall_since = now
                    elif now - stall_since >= 8.0:
                        self.get_logger().warn(
                            "NavigateToPosition stalled on heading "
                            f"(remaining_angle={ang:.3f}rad) — canceling"
                        )
                        goal_handle.cancel_goal_async()
                        # Treat as soft failure so caller can re-sense / face.
                        return 1
                else:
                    stall_since = None
                last_angle = ang
            time.sleep(0.05)

        result = result_future.result()
        if result is None:
            self.get_logger().error("No navigate result")
            return 1

        status = result.status
        # action_msgs/GoalStatus: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED
        status_name = {
            4: "SUCCEEDED",
            5: "CANCELED",
            6: "ABORTED",
        }.get(status, str(status))
        if status != 4:
            hint = ""
            if status == 6:
                hint = (
                    " (Create3 abort: usually cliff/bump/stall/hazard — "
                    "check /create3/hazard_detection; path must be clear)"
                )
            self.get_logger().error(
                f"NavigateToPosition finished with status={status} "
                f"({status_name}){hint}"
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
        self._last_nav_fb = fb
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
