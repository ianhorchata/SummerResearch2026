"""Lawn-mower sweep of a rectangular carpet with pick interrupts + auto-dock.

Flow: Undock → establish area frame from odom → boustrophedon lanes with
periodic vision scans → pick_one() when an object is seen → return to the
same waypoint and re-scan before advancing → drive near the dock corner
facing the charger → Dock when coverage finishes, battery is low, or
servo voltage stays below threshold for a sustained hold (default 10 s —
ignores brief grab brownouts).

Each lane runs forward/back along the 9 ft length; lanes step across
the 5 ft width (lawn-mower along the long axis).

    ros2 run robot_vision sweep_and_pick
    ros2 launch robot_vision sweep_and_pick.launch.py

Assumes robot_bringup (cameras, arm, vision) and Create3 are already up.

Dock geometry: while docked the Create3 faces **into** the dock; Undock
turns ~180° so the robot faces out into the carpet. Area origin XY is
frozen at the dock corner; the long axis uses the **post-undock** heading
(not the docked heading). Set ``area_yaw_offset_rad`` if needed.

Assumes the charging dock is in the **bottom-left** corner of the carpet:
9 ft forward along the post-undock heading, 5 ft to the **right**.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from irobot_create_msgs.action import Dock, DriveDistance, RotateAngle, Undock
from sensor_msgs.msg import BatteryState

from robot_interfaces.msg import ServoHealth
from robot_vision.drive_to_object import DriveToObject

_FT = 0.3048
_DEFAULT_LENGTH_M = 10.0 * _FT  # 2.7432
_DEFAULT_WIDTH_M = 6.0 * _FT   # 1.5240
_HALF_PI = math.pi * 0.5


class SweepAndPick(DriveToObject):
    def __init__(self):
        # Parent declares pick/drive params and creates odom/detect/nav clients.
        super().__init__(node_name="sweep_and_pick")

        self.declare_parameter("area_length_m", _DEFAULT_LENGTH_M)
        self.declare_parameter("area_width_m", _DEFAULT_WIDTH_M)
        # Keep robot center inside carpet; Create3 radius ~0.17m + buffer.
        self.declare_parameter("margin_m", 0.35)
        self.declare_parameter("lane_spacing_m", 0.30)
        self.declare_parameter("scan_interval_m", 0.40)
        self.declare_parameter("clear_dock_distance_m", 0.30)
        # How far into the carpet (from dock origin) to stop before Dock.
        # Create3 IR docking needs the robot nearby and facing the charger.
        self.declare_parameter("pre_dock_approach_m", 0.70)
        self.declare_parameter("area_yaw_offset_rad", 0.0)
        self.declare_parameter("battery_low_pct", 0.20)
        self.declare_parameter("servo_low_voltage_v", 7.8)
        # Ignore brief brownouts during grab/close; only dock if still low.
        self.declare_parameter("servo_low_hold_sec", 10.0)
        self.declare_parameter("max_picks", 20)
        self.declare_parameter("drive_timeout_sec", 120.0)
        self.declare_parameter("dock_timeout_sec", 180.0)
        self.declare_parameter("battery_topic", "/create3/battery_state")
        self.declare_parameter("arm_health_topic", "arm/health")
        self.declare_parameter("undock_action", "/create3/undock")
        self.declare_parameter("dock_action", "/create3/dock")
        self.declare_parameter("drive_distance_action", "/create3/drive_distance")
        self.declare_parameter("rotate_angle_action", "/create3/rotate_angle")
        self.declare_parameter("skip_undock", False)
        self.declare_parameter("skip_dock", False)
        # If True, refuse pick targets whose ground pose is outside the area.
        self.declare_parameter("pick_only_inside_area", True)
        # Ignore detections near area origin (dock sits in that corner).
        self.declare_parameter("dock_keepout_m", 0.90)

        self._area_length = float(self.get_parameter("area_length_m").value)
        self._area_width = float(self.get_parameter("area_width_m").value)
        self._lane_spacing = float(self.get_parameter("lane_spacing_m").value)
        self._margin = float(self.get_parameter("margin_m").value)
        self._scan_interval = float(self.get_parameter("scan_interval_m").value)
        self._clear_dock = float(
            self.get_parameter("clear_dock_distance_m").value)
        self._pre_dock_approach = max(
            0.20, float(self.get_parameter("pre_dock_approach_m").value)
        )
        self._dock_keepout = float(
            self.get_parameter("dock_keepout_m").value)
        self._yaw_offset = float(
            self.get_parameter("area_yaw_offset_rad").value)
        self._battery_low = float(self.get_parameter("battery_low_pct").value)
        self._servo_low_v = float(
            self.get_parameter("servo_low_voltage_v").value)
        self._servo_low_hold = float(
            self.get_parameter("servo_low_hold_sec").value)
        self._max_picks = int(self.get_parameter("max_picks").value)
        self._drive_timeout = float(
            self.get_parameter("drive_timeout_sec").value)
        self._dock_timeout = float(
            self.get_parameter("dock_timeout_sec").value)
        self._skip_undock = self._as_bool(
            self.get_parameter("skip_undock").value)
        self._skip_dock = self._as_bool(self.get_parameter("skip_dock").value)
        self._pick_only_inside = self._as_bool(
            self.get_parameter("pick_only_inside_area").value)

        # Origin XY = dock corner (captured while docked). Length/width yaw
        # come from the post-undock heading (docked face is 180° opposite).
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._length_yaw = 0.0
        self._width_yaw = 0.0

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._battery_pct: Optional[float] = None
        self._min_servo_v: Optional[float] = None
        # monotonic time when servo voltage first went below threshold; None = OK.
        self._servo_low_since: Optional[float] = None
        self._abort_reason: Optional[str] = None
        self._pick_count = 0
        # Odom XY of targets we refuse (outside / unreachable standoff).
        # Consensus ignores detections near these so we can pick another object.
        self._skip_odom: list = []
        self._skip_gate_m = 0.30

        self.create_subscription(
            BatteryState,
            str(self.get_parameter("battery_topic").value),
            self._on_battery,
            sensor_qos,
            callback_group=self._cb_group,
        )
        self.create_subscription(
            ServoHealth,
            str(self.get_parameter("arm_health_topic").value),
            self._on_arm_health,
            10,
            callback_group=self._cb_group,
        )

        self._undock_cli = ActionClient(
            self,
            Undock,
            str(self.get_parameter("undock_action").value),
            callback_group=self._cb_group,
        )
        self._dock_cli = ActionClient(
            self,
            Dock,
            str(self.get_parameter("dock_action").value),
            callback_group=self._cb_group,
        )
        self._drive_cli = ActionClient(
            self,
            DriveDistance,
            str(self.get_parameter("drive_distance_action").value),
            callback_group=self._cb_group,
        )
        self._rotate_cli = ActionClient(
            self,
            RotateAngle,
            str(self.get_parameter("rotate_angle_action").value),
            callback_group=self._cb_group,
        )

        usable_w = max(0.0, self._area_width - 2.0 * self._margin)
        usable_l = max(0.0, self._area_length - 2.0 * self._margin)
        n_lanes = 1
        if self._lane_spacing > 1e-3 and usable_w > 0.0:
            n_lanes = int(math.floor(usable_w / self._lane_spacing)) + 1
        self.get_logger().info(
            f"sweep_and_pick ready: area={self._area_length:.3f}x"
            f"{self._area_width:.3f}m margin={self._margin:.2f}m "
            f"dock_keepout={self._dock_keepout:.2f}m "
            f"lanes≈{n_lanes} spacing={self._lane_spacing:.2f}m "
            f"scan_every={self._scan_interval:.2f}m "
            f"battery_low={self._battery_low:.0%} "
            f"servo_low={self._servo_low_v:.1f}V "
            f"hold={self._servo_low_hold:.0f}s "
            f"max_picks={self._max_picks}"
        )

    # -------------------------------------------------------------- health

    def _on_battery(self, msg: BatteryState) -> None:
        pct = float(msg.percentage)
        # Create3 may publish 0–1 or 0–100; normalize to 0–1.
        if pct > 1.0:
            pct = pct / 100.0
        self._battery_pct = pct
        if pct <= self._battery_low and self._abort_reason is None:
            self._abort_reason = (
                f"battery {pct:.0%} <= {self._battery_low:.0%}"
            )
            self.get_logger().warn(f"Auto-dock trigger: {self._abort_reason}")

    def _on_arm_health(self, msg: ServoHealth) -> None:
        voltages = [float(v) for v in msg.voltage if float(v) > 0.0]
        if not voltages:
            return
        self._min_servo_v = min(voltages)
        if self._min_servo_v >= self._servo_low_v:
            if self._servo_low_since is not None:
                held = time.monotonic() - self._servo_low_since
                self.get_logger().info(
                    f"Servo voltage recovered to {self._min_servo_v:.2f}V "
                    f"after {held:.1f}s below {self._servo_low_v:.1f}V "
                    f"(need {self._servo_low_hold:.0f}s for auto-dock)"
                )
            self._servo_low_since = None
            return

        now = time.monotonic()
        if self._servo_low_since is None:
            self._servo_low_since = now
            self.get_logger().warn(
                f"Servo voltage {self._min_servo_v:.2f}V "
                f"< {self._servo_low_v:.1f}V — holding "
                f"{self._servo_low_hold:.0f}s before auto-dock"
            )
            return

        held = now - self._servo_low_since
        if held < self._servo_low_hold or self._abort_reason is not None:
            return

        self._abort_reason = (
            f"servo voltage {self._min_servo_v:.2f}V "
            f"< {self._servo_low_v:.1f}V for {held:.1f}s"
        )
        self.get_logger().warn(f"Auto-dock trigger: {self._abort_reason}")

    def _should_abort_mission(self) -> bool:
        return self._abort_reason is not None

    # -------------------------------------------------------------- mission

    def run_mission(self) -> int:
        if not self._wait_for_odom(10.0):
            self.get_logger().error("No odom — is Create 3 up?")
            return 1
        if not self._detect_cli.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("/vision/detect_poses not available")
            return 1

        if not self._skip_undock:
            # XY at dock corner (full 9x5 from the wall), but do NOT use
            # docked yaw — while docked the robot faces into the dock;
            # Undock turns ~180° to face the carpet.
            if self._odom is None:
                self.get_logger().error("No odom before undock")
                return 1
            self._capture_area_origin_xy("pre-undock dock corner")
            self.get_logger().info("===== UNDOCK =====")
            if not self._undock():
                return 1
            if self._should_abort_mission():
                return self._finish_dock()
        else:
            self.get_logger().info(
                "skip_undock=true — origin+yaw from current pose "
                "(assume already facing into carpet)"
            )
            if self._odom is None:
                self.get_logger().error("No odom")
                return 1
            self._capture_area_origin_xy("skip_undock")
            self._capture_area_yaw("skip_undock (facing carpet)")

        self.get_logger().info("===== ESTABLISH AREA =====")
        if not self._settle_before_sense():
            self.get_logger().warn("Could not settle after undock — continuing")

        if not self._skip_undock:
            # After Undock's 180° turn: +X = into carpet, +Y = right.
            if self._odom is None:
                self.get_logger().error("Lost odom after undock")
                return 1
            self._capture_area_yaw("post-undock (facing carpet)")

        waypoints = self._build_lawnmower_waypoints()
        if not waypoints:
            self.get_logger().error("No sweep waypoints — check area/margin params")
            return 1
        self.get_logger().info(
            f"===== SWEEP {len(waypoints)} waypoints "
            f"(odom-clamped lawn-mower, margin={self._margin:.2f}m) ====="
        )

        for i, (ax, ay, yaw) in enumerate(waypoints):
            if self._should_abort_mission():
                return self._finish_dock()
            if self._pick_count >= self._max_picks:
                self.get_logger().warn(
                    f"max_picks={self._max_picks} reached — docking"
                )
                return self._finish_dock()

            self.get_logger().info(
                f"----- Waypoint {i + 1}/{len(waypoints)} "
                f"area=({ax:.3f}, {ay:.3f}) -----"
            )
            nav = self._navigate_to_area_pose(ax, ay, yaw)
            if nav == 2 or self._should_abort_mission():
                return self._finish_dock()
            if nav != 0:
                self.get_logger().warn(
                    "Navigate to waypoint failed — skipping to next"
                )
                continue

            # Stay inside: if odom drifted outside, snap back before scanning.
            if not self._robot_inside_area(slack=0.05):
                self.get_logger().warn(
                    "Robot outside inset area after nav — "
                    "re-navigating to clamped pose"
                )
                cx, cy = self._clamp_area_xy(*self._odom_to_area_xy())
                if self._navigate_to_area_pose(cx, cy, yaw) != 0:
                    continue

            # Scan → pick → return → re-scan at this waypoint before advancing.
            # Nearby objects are often still visible after a pick; don't skip
            # them by driving to the next node immediately.
            while True:
                if self._should_abort_mission():
                    return self._finish_dock()
                if self._pick_count >= self._max_picks:
                    self.get_logger().warn(
                        f"max_picks={self._max_picks} reached — docking"
                    )
                    return self._finish_dock()

                if not self.scan_for_pickable():
                    break

                if self._pick_only_inside and not self._best_object_inside_area():
                    self.get_logger().info(
                        "Nearest object is outside carpet bounds / "
                        "standoff unreachable — skip and look for another"
                    )
                    self._remember_skip_current_lock()
                    continue

                self.get_logger().info(
                    f"Object seen — pick_one "
                    f"({self._pick_count + 1}/{self._max_picks})"
                )
                # pick_one clears the lock in finally — keep a copy for blacklist.
                locked_before = self._lock_odom_xy
                code = self.pick_one()
                if code == 2 or self._should_abort_mission():
                    return self._finish_dock()
                if code != 0:
                    # Failed picks (edge/unreachable, max iters, etc.) often
                    # re-lock the same target — blacklist it and look for another
                    # at this waypoint instead of looping forever.
                    self.get_logger().warn(
                        "Pick failed — skipping this target, re-scanning"
                    )
                    if locked_before is not None:
                        self._lock_odom_xy = locked_before
                    self._remember_skip_current_lock()
                    nav_back = self._navigate_to_area_pose(ax, ay, yaw)
                    if nav_back == 2 or self._should_abort_mission():
                        return self._finish_dock()
                    if nav_back != 0:
                        break
                    continue

                self._pick_count += 1
                self.get_logger().info(
                    f"Pick succeeded (total={self._pick_count})"
                )

                # Snap back onto the planned path, then re-scan here.
                self.get_logger().info(
                    f"Returning to sweep waypoint area=({ax:.3f}, {ay:.3f}) "
                    "— will re-scan before advancing"
                )
                nav_back = self._navigate_to_area_pose(ax, ay, yaw)
                if nav_back == 2 or self._should_abort_mission():
                    return self._finish_dock()
                if nav_back != 0:
                    self.get_logger().warn(
                        "Return to waypoint failed — advancing sweep"
                    )
                    break

        self.get_logger().info(
            f"===== COVERAGE COMPLETE (picks={self._pick_count}) ====="
        )
        return self._finish_dock()

    # -------------------------------------------------------- area geometry

    def _capture_area_origin_xy(self, label: str) -> None:
        """Freeze carpet origin at current odom XY (dock corner when docked)."""
        op = self._odom.pose.pose.position
        self._origin_x = float(op.x)
        self._origin_y = float(op.y)
        self.get_logger().info(
            f"Area origin XY [{label}] odom=({self._origin_x:.3f}, "
            f"{self._origin_y:.3f}) "
            f"bounds length=[0,{self._area_length:.3f}] "
            f"width=[0,{self._area_width:.3f}]"
        )

    def _capture_area_yaw(self, label: str) -> None:
        """Set +X/+Y from current heading. Call after Undock (faces carpet).

        Docked heading points into the dock; post-undock is ~180° opposite.
        +X = forward along that heading (9 ft), +Y = right (5 ft).
        """
        oq = self._odom.pose.pose.orientation
        current_yaw = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
        self._length_yaw = self._wrap_pi(current_yaw + self._yaw_offset)
        self._width_yaw = self._wrap_pi(self._length_yaw - _HALF_PI)
        self.get_logger().info(
            f"Area axes [{label}] "
            f"length_yaw={self._length_yaw:.3f}rad "
            f"({math.degrees(self._length_yaw):.1f}deg forward), "
            f"width_yaw={self._width_yaw:.3f}rad "
            f"({math.degrees(self._width_yaw):.1f}deg right)"
        )

    def _build_lawnmower_waypoints(self):
        """Scan poses in area frame, all inside [margin, L-margin] x [margin, W-margin].

        Lanes step across the short (width) axis; each lane drives forward/back
        along the long (length) axis so scans run the full 9 ft carpet depth.
        """
        m = self._margin
        x0, x1 = m, max(m, self._area_length - m)
        y0, y1 = m, max(m, self._area_width - m)
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            return []

        ys = [y0]
        if self._lane_spacing > 1e-3:
            y = y0 + self._lane_spacing
            while y < y1 - 1e-6:
                ys.append(y)
                y += self._lane_spacing
            if ys[-1] < y1 - 0.05:
                ys.append(y1)
        else:
            ys.append(y1)

        waypoints = []
        for lane_i, ay in enumerate(ys):
            going_forward = (lane_i % 2 == 0)
            yaw = (
                self._length_yaw if going_forward
                else self._length_yaw + math.pi
            )
            if going_forward:
                xs = self._sample_axis(x0, x1, self._scan_interval)
            else:
                xs = self._sample_axis(x1, x0, self._scan_interval)
            for ax in xs:
                waypoints.append((ax, ay, yaw))
        return waypoints

    @staticmethod
    def _sample_axis(start: float, end: float, step: float):
        """Inclusive samples from start→end with approx ``step`` spacing.

        Must include ``start`` so lane changes visit the corner
        (ax_new, y_end) before driving the next row. Skipping start makes
        NavigateToPosition cut diagonally to the first mid-lane scan —
        a sawtooth with missing corners.
        """
        span = end - start
        if abs(span) < 1e-6:
            return [start]
        step = max(0.05, abs(step))
        n = max(1, int(math.ceil(abs(span) / step)))
        return [start + span * (i / n) for i in range(0, n + 1)]

    def _area_to_odom_xy(self, ax: float, ay: float):
        c_l, s_l = math.cos(self._length_yaw), math.sin(self._length_yaw)
        c_w, s_w = math.cos(self._width_yaw), math.sin(self._width_yaw)
        ox = self._origin_x + ax * c_l + ay * c_w
        oy = self._origin_y + ax * s_l + ay * s_w
        return ox, oy

    def _area_to_odom(self, ax: float, ay: float, yaw: float):
        from geometry_msgs.msg import PoseStamped

        ox, oy = self._area_to_odom_xy(ax, ay)
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "odom"
        goal.pose.position.x = ox
        goal.pose.position.y = oy
        goal.pose.position.z = 0.0
        goal.pose.orientation = self._quat_from_yaw(yaw)
        return goal

    def _odom_xy_to_area(self, ox: float, oy: float):
        dx = float(ox) - self._origin_x
        dy = float(oy) - self._origin_y
        c_l, s_l = math.cos(self._length_yaw), math.sin(self._length_yaw)
        c_w, s_w = math.cos(self._width_yaw), math.sin(self._width_yaw)
        det = c_l * s_w - c_w * s_l
        if abs(det) < 1e-9:
            return 0.0, 0.0
        ax = (dx * s_w - dy * c_w) / det
        ay = (c_l * dy - s_l * dx) / det
        return ax, ay

    def _odom_to_area_xy(self):
        if self._odom is None:
            return 0.0, 0.0
        op = self._odom.pose.pose.position
        return self._odom_xy_to_area(float(op.x), float(op.y))

    def _clamp_area_xy(self, ax: float, ay: float):
        m = self._margin
        ax = min(max(ax, m), self._area_length - m)
        ay = min(max(ay, m), self._area_width - m)
        return ax, ay

    def _robot_inside_area(self, slack: float = 0.0) -> bool:
        ax, ay = self._odom_to_area_xy()
        m = max(0.0, self._margin - slack)
        return (
            m <= ax <= self._area_length - m
            and m <= ay <= self._area_width - m
        )

    def _clamp_odom_goal(self, goal_pose):
        """Keep NavigateToPosition goals inside the inset carpet rectangle.

        Returns (pose, clamp_delta_m). Waypoint goals are pre-clamped so
        delta is ~0; approach goals that need to leave the carpet get a
        large delta and should abort the pick.
        """
        from copy import deepcopy

        gx = float(goal_pose.pose.position.x)
        gy = float(goal_pose.pose.position.y)
        ax, ay = self._odom_xy_to_area(gx, gy)
        cx, cy = self._clamp_area_xy(ax, ay)
        if abs(cx - ax) < 1e-4 and abs(cy - ay) < 1e-4:
            return goal_pose, 0.0
        ox, oy = self._area_to_odom_xy(cx, cy)
        clamp_delta = math.hypot(cx - ax, cy - ay)
        clamped = deepcopy(goal_pose)
        clamped.pose.position.x = ox
        clamped.pose.position.y = oy
        self.get_logger().warn(
            f"Clamped nav goal area ({ax:.3f},{ay:.3f}) -> "
            f"({cx:.3f},{cy:.3f}) delta={clamp_delta:.3f}m to stay in bounds"
        )
        return clamped, clamp_delta

    def _navigate(self, goal_pose) -> int:
        """Override: keep goals on carpet; abort if approach needs to leave."""
        clamped, delta = self._clamp_odom_goal(goal_pose)
        # Waypoints are pre-clamped (delta≈0). A large clamp means the
        # approach standoff is off-carpet — stop instead of looping on the edge.
        if delta > max(0.05, self._approach_tol):
            self.get_logger().warn(
                f"Approach goal outside area by {delta:.3f}m — aborting pick"
            )
            return 1
        return super()._navigate(clamped)

    def _navigate_to_area_pose(self, ax: float, ay: float, yaw: float) -> int:
        ax, ay = self._clamp_area_xy(ax, ay)
        goal = self._area_to_odom(ax, ay, yaw)
        self._goal_pub.publish(goal)
        self.get_logger().info(
            f"Nav area=({ax:.3f},{ay:.3f}) -> "
            f"odom=({goal.pose.position.x:.3f},{goal.pose.position.y:.3f})"
        )
        return self._navigate(goal)

    def _xy_inside_inset(self, ax: float, ay: float) -> bool:
        m = self._margin
        return (
            m <= ax <= self._area_length - m
            and m <= ay <= self._area_width - m
        )

    def _remember_skip_current_lock(self) -> None:
        """Blacklist the current lock so consensus will not re-chase it."""
        if self._lock_odom_xy is not None:
            sx, sy = self._lock_odom_xy
            self._skip_odom.append((float(sx), float(sy)))
            self.get_logger().info(
                f"Blacklisted target odom=({sx:.3f}, {sy:.3f}) "
                f"(skip_gate={self._skip_gate_m:.2f}m, "
                f"n_skipped={len(self._skip_odom)})"
            )
        self._clear_target_lock()
        self._last_scan_pose = None

    def _is_skipped_odom(self, ox: float, oy: float) -> bool:
        for sx, sy in self._skip_odom:
            if math.hypot(ox - sx, oy - sy) <= self._skip_gate_m:
                return True
        return False

    def _is_dock_keepout(self, ox: float, oy: float) -> bool:
        """True if odom XY falls near the dock corner (area origin)."""
        if self._dock_keepout <= 0.0:
            return False
        ax, ay = self._odom_xy_to_area(ox, oy)
        return math.hypot(ax, ay) <= self._dock_keepout

    def _valid_candidates(
        self,
        detections,
        poses,
        widths_m,
        heights_m,
        match_costs,
        *,
        relax: bool = False,
    ):
        cands = super()._valid_candidates(
            detections,
            poses,
            widths_m,
            heights_m,
            match_costs,
            relax=relax,
        )
        if not cands:
            return cands
        kept = []
        n_dock = 0
        n_skip = 0
        for c in cands:
            ox = float(c["ox"])
            oy = float(c["oy"])
            if self._is_dock_keepout(ox, oy):
                n_dock += 1
                continue
            if self._is_skipped_odom(ox, oy):
                n_skip += 1
                continue
            kept.append(c)
        if n_dock > 0:
            self.get_logger().info(
                f"Filtered {n_dock} detection(s) in dock keep-out "
                f"({self._dock_keepout:.2f}m from origin)"
            )
        if n_skip > 0:
            self.get_logger().info(
                f"Filtered {n_skip} blacklisted detection(s)"
            )
        return kept

    def _best_object_inside_area(self) -> bool:
        """True if object and approach standoff both lie inside the carpet.

        Reuses ``_last_scan_pose`` from ``scan_for_pickable`` — no second
        ``detect_poses`` (robot has not moved yet).

        Object-only checks are not enough: planar approach stands
        ``grab_distance_m`` back, and that standoff is often past the margin
        for edge objects. Picking those clamps nav goals and loops forever.
        """
        pose = self._last_scan_pose
        if pose is None:
            self.get_logger().warn(
                "No cached scan pose for bounds check — skipping pick"
            )
            return False
        if self._odom is None:
            return False
        ox_b = float(pose.pose.position.x)
        oy_b = float(pose.pose.position.y)
        op = self._odom.pose.pose.position
        oq = self._odom.pose.pose.orientation
        yaw_r = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
        c, s = math.cos(yaw_r), math.sin(yaw_r)
        ox = float(op.x) + c * ox_b - s * oy_b
        oy = float(op.y) + s * ox_b + c * oy_b
        ax, ay = self._odom_xy_to_area(ox, oy)
        obj_inside = self._xy_inside_inset(ax, ay)

        # Approach standoff in base_link, then to area frame.
        approach = self._approach_in_base_link(ox_b, oy_b)
        stand_inside = True
        sax = say = float("nan")
        if approach is None:
            stand_inside = False
        else:
            gx_b, gy_b, _yaw, already_close = approach
            if already_close:
                # Already at/inside grab range — robot pose is the standoff.
                sax, say = self._odom_to_area_xy()
            else:
                gx = float(op.x) + c * gx_b - s * gy_b
                gy = float(op.y) + s * gx_b + c * gy_b
                sax, say = self._odom_xy_to_area(gx, gy)
            stand_inside = self._xy_inside_inset(sax, say)

        inside = obj_inside and stand_inside
        self.get_logger().info(
            f"Object area=({ax:.3f},{ay:.3f}) "
            f"{'INSIDE' if obj_inside else 'OUTSIDE'} inset; "
            f"standoff area=({sax:.3f},{say:.3f}) "
            f"{'INSIDE' if stand_inside else 'OUTSIDE'} inset"
            + ("" if inside else " — skip pick")
        )
        return inside

    def _approach_dock(self) -> bool:
        """Drive near the dock corner, facing the charger, before Dock.

        Area origin is the dock corner; +X faces into the carpet. Approach
        at (pre_dock_approach_m, margin) facing −X so Create3 IR can see
        the dock from a short, reliable range.
        """
        ax = max(self._margin, self._pre_dock_approach)
        ay = self._margin
        face_dock_yaw = self._wrap_pi(self._length_yaw + math.pi)
        cur_ax, cur_ay = self._odom_to_area_xy()
        dist = math.hypot(cur_ax - ax, cur_ay - ay)
        self.get_logger().info(
            f"===== APPROACH DOCK "
            f"area=({ax:.3f},{ay:.3f}) face_yaw={face_dock_yaw:.3f} "
            f"(from area=({cur_ax:.3f},{cur_ay:.3f}), dist={dist:.2f}m) ====="
        )
        nav = self._navigate_to_area_pose(ax, ay, face_dock_yaw)
        if nav != 0:
            self.get_logger().warn(
                f"Pre-dock approach nav failed (code={nav}) — "
                "attempting Dock from current pose"
            )
            return False
        return True

    def _finish_dock(self) -> int:
        reason = self._abort_reason or "mission complete"
        if self._skip_dock:
            self.get_logger().info(
                f"skip_dock=true — not docking ({reason})"
            )
            return 0 if self._abort_reason is None else 2
        self._approach_dock()
        self.get_logger().info(f"===== DOCK ({reason}) =====")
        ok = self._dock()
        return 0 if ok and self._abort_reason is None else (0 if ok else 1)

    # -------------------------------------------------------- Create3 actions

    def _undock(self) -> bool:
        return self._run_empty_action(
            self._undock_cli, Undock.Goal(), "Undock", self._dock_timeout
        )

    def _dock(self) -> bool:
        return self._run_empty_action(
            self._dock_cli, Dock.Goal(), "Dock", self._dock_timeout
        )

    def _drive_distance(self, distance_m: float) -> bool:
        if abs(distance_m) < 1e-3:
            return True
        if self._should_abort_mission():
            return False
        if not self._drive_cli.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("DriveDistance action server not available")
            return False
        goal = DriveDistance.Goal()
        goal.distance = float(distance_m)
        goal.max_translation_speed = float(self._max_tx)
        self.get_logger().info(f"DriveDistance {distance_m:.3f}m")
        return self._wait_action_result(
            self._drive_cli, goal, "DriveDistance", self._drive_timeout
        )

    def _rotate_angle(self, angle_rad: float) -> bool:
        if abs(angle_rad) < 1e-3:
            return True
        if self._should_abort_mission():
            return False
        if not self._rotate_cli.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("RotateAngle action server not available")
            return False
        goal = RotateAngle.Goal()
        goal.angle = float(angle_rad)
        goal.max_rotation_speed = float(self._max_rot)
        self.get_logger().info(
            f"RotateAngle {angle_rad:.3f}rad "
            f"({math.degrees(angle_rad):.1f}deg)"
        )
        return self._wait_action_result(
            self._rotate_cli, goal, "RotateAngle", self._drive_timeout
        )

    def _rotate_to_yaw(self, target_yaw: float) -> bool:
        """Rotate in place to an absolute odom yaw."""
        if self._odom is None:
            return False
        oq = self._odom.pose.pose.orientation
        current = self._yaw_from_quat(oq.x, oq.y, oq.z, oq.w)
        delta = self._wrap_pi(target_yaw - current)
        return self._rotate_angle(delta)

    def _run_empty_action(
        self, client: ActionClient, goal, name: str, timeout_sec: float
    ) -> bool:
        if not client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(f"{name} action server not available")
            return False
        self.get_logger().info(f"Sending {name} goal...")
        return self._wait_action_result(client, goal, name, timeout_sec)

    def _wait_action_result(
        self, client: ActionClient, goal, name: str, timeout_sec: float
    ) -> bool:
        send_future = client.send_goal_async(goal)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not send_future.done():
            if self._should_abort_mission() and name not in ("Dock", "Undock"):
                self.get_logger().warn(f"Abort while waiting for {name} accept")
                return False
            if time.monotonic() > deadline:
                self.get_logger().error(f"{name}: timed out waiting for accept")
                return False
            time.sleep(0.05)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"{name} goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if self._should_abort_mission() and name not in ("Dock", "Undock"):
                self.get_logger().warn(f"Abort — canceling {name}")
                goal_handle.cancel_goal_async()
                return False
            if time.monotonic() > deadline:
                self.get_logger().error(f"{name} timed out")
                goal_handle.cancel_goal_async()
                return False
            time.sleep(0.05)

        result = result_future.result()
        if result is None:
            self.get_logger().error(f"{name}: no result")
            return False
        # action_msgs/GoalStatus: 4 = SUCCEEDED
        if result.status != 4:
            self.get_logger().error(
                f"{name} finished with status={result.status}"
            )
            return False
        self.get_logger().info(f"{name} succeeded")
        return True

    @staticmethod
    def _wrap_pi(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = SweepAndPick()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        code = node.run_mission()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
