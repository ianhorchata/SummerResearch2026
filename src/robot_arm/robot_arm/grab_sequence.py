"""Arm grab sequence (taught drop + optional IK grab).

Sequence
--------
1. Move to the grab pose (Servo1–4).
2. Close the gripper (Servo4).
3. Move to the drop-in-basket pose (Servo1–3; gripper stays closed).
4. Open the gripper (Servo4).

Uses ``/arm/move_joints`` so each motion can set travel time (default 2 s).
Poses are in radians (same as ``/arm/joint_states``) and converted through
the same joint calibration as ``arm.launch.py``.

``drive_to_object`` can replace ``grab_pose`` with planar IK from stereo
(``robot_arm.planar_ik``); ``DEFAULT_DROP`` stays taught.

    ros2 run robot_arm grab_sequence
    ros2 run robot_arm grab_sequence --ros-args -p dry_run:=true
    ros2 run robot_arm grab_sequence --ros-args -p move_ms:=2000
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from robot_interfaces.srv import MoveJoints

from .hiwonder_servo import DEG_PER_UNIT, POS_MAX, POS_MIN

_RAD_PER_UNIT = math.radians(DEG_PER_UNIT)

# Calibrated poses from teach pendant / joint_states (radians).
DEFAULT_GRAB = [1.32, 1.0890854532444616, -0.9131562646434331, -1.0974630336540343]
DEFAULT_DROP = [0.16755160819145562, -0.975988117715229, -1.7551030958054976]
# Servo4 center in arm.launch.py maps to 0.0 rad = gripper closed.
DEFAULT_GRIPPER_CLOSED = 0.0
DEFAULT_GRIPPER_OPEN = -1.0807078728348887
DEFAULT_MOVE_MS = 2000

# Must match arm.launch.py joint_name_map: id -> (center_units, sign).
_DEFAULT_CALIB: Dict[int, Tuple[int, int]] = {
    1: (481, -1),
    2: (488, -1),
    3: (531, -1),
    4: (900, 1),
}

_NAME_TO_ID = {"Servo1": 1, "Servo2": 2, "Servo3": 3, "Servo4": 4}


def _rad_to_abs_deg(servo_id: int, rad: float,
                    calib: Optional[Dict[int, Tuple[int, int]]] = None) -> float:
    """Convert calibrated radians -> absolute degrees for MoveJoints."""
    center, sign = (calib or _DEFAULT_CALIB)[servo_id]
    units = int(round(center + (rad / (_RAD_PER_UNIT * sign))))
    units = max(POS_MIN, min(POS_MAX, units))
    return units * DEG_PER_UNIT


def run_grab_sequence(
    node: Node,
    move_cli,
    *,
    grab_pose: Sequence[float] = DEFAULT_GRAB,
    drop_pose: Sequence[float] = DEFAULT_DROP,
    gripper_closed_rad: float = DEFAULT_GRIPPER_CLOSED,
    gripper_open_rad: float = DEFAULT_GRIPPER_OPEN,
    move_ms: int = DEFAULT_MOVE_MS,
    dry_run: bool = False,
    timeout_extra_sec: float = 5.0,
) -> None:
    """Execute the 4-step grab on an existing node + MoveJoints client.

    Relies on an executor already spinning ``node`` (or a single-threaded
    caller that can afford blocking). Polls service futures with sleep.
    Raises RuntimeError on failure.
    """
    grab = [float(x) for x in grab_pose]
    drop = [float(x) for x in drop_pose]
    if len(grab) != 4:
        raise ValueError(f"grab_pose must have 4 values, got {len(grab)}")
    if len(drop) != 3:
        raise ValueError(f"drop_pose must have 3 values, got {len(drop)}")
    if move_ms <= 0:
        raise ValueError(f"move_ms must be > 0, got {move_ms}")

    log = node.get_logger()
    log.info(
        f"grab sequence start (move_ms={move_ms})"
        + (" dry_run" if dry_run else "")
    )

    def move(names: Sequence[str], positions_rad: Sequence[float], label: str) -> None:
        ids: List[int] = []
        degs: List[float] = []
        for name, rad in zip(names, positions_rad):
            sid = _NAME_TO_ID[name]
            ids.append(sid)
            degs.append(_rad_to_abs_deg(sid, float(rad)))

        pretty = ", ".join(
            f"{n}={r:.4f}rad({d:.1f}deg)"
            for n, r, d in zip(names, positions_rad, degs)
        )
        if dry_run:
            log.info(f"[dry_run] {label} over {move_ms}ms: {pretty}")
            return

        req = MoveJoints.Request()
        req.ids = ids
        req.positions_deg = degs
        req.time_ms = move_ms
        future = move_cli.call_async(req)

        deadline = time.monotonic() + (move_ms / 1000.0) + timeout_extra_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f"{label}: timed out waiting for /arm/move_joints")
            time.sleep(0.05)

        result = future.result()
        if result is None or not result.success:
            msg = getattr(result, "message", "no response")
            raise RuntimeError(f"{label}: move_joints failed: {msg}")

        log.info(f"{label} over {move_ms}ms: {pretty}")
        time.sleep(move_ms / 1000.0)

    move(["Servo1", "Servo2", "Servo3", "Servo4"], grab, "1/4 grab pose")
    move(["Servo4"], [float(gripper_closed_rad)], "2/4 close gripper")
    move(["Servo1", "Servo2", "Servo3"], drop, "3/4 drop pose")
    # Close/drop often browns out the servo bus; MoveJoints returns success as
    # soon as the command is queued, so a single open can leave jaws shut.
    # Pause briefly for voltage recovery, then open twice.
    time.sleep(0.5)
    move(["Servo4"], [float(gripper_open_rad)], "4/4 open gripper")
    time.sleep(0.3)
    move(["Servo4"], [float(gripper_open_rad)], "4/4 open gripper (retry)")
    log.info("grab sequence done")


class GrabSequence(Node):
    def __init__(self) -> None:
        super().__init__("grab_sequence")

        self.declare_parameter("grab_pose", DEFAULT_GRAB)
        self.declare_parameter("drop_pose", DEFAULT_DROP)
        self.declare_parameter("gripper_closed_rad", DEFAULT_GRIPPER_CLOSED)
        self.declare_parameter("gripper_open_rad", DEFAULT_GRIPPER_OPEN)
        self.declare_parameter("move_ms", DEFAULT_MOVE_MS)
        self.declare_parameter("move_joints_service", "/arm/move_joints")
        self.declare_parameter("dry_run", False)

        self._grab = [float(x) for x in self.get_parameter("grab_pose").value]
        self._drop = [float(x) for x in self.get_parameter("drop_pose").value]
        self._closed = float(self.get_parameter("gripper_closed_rad").value)
        self._open = float(self.get_parameter("gripper_open_rad").value)
        self._move_ms = int(self.get_parameter("move_ms").value)
        dry = self.get_parameter("dry_run").value
        if isinstance(dry, str):
            self._dry_run = dry.strip().lower() in ("1", "true", "yes", "on")
        else:
            self._dry_run = bool(dry)

        svc = str(self.get_parameter("move_joints_service").value)
        self._cli = self.create_client(MoveJoints, svc)
        if not self._dry_run and not self._cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"service {svc} not available")

    def run(self) -> None:
        run_grab_sequence(
            self,
            self._cli,
            grab_pose=self._grab,
            drop_pose=self._drop,
            gripper_closed_rad=self._closed,
            gripper_open_rad=self._open,
            move_ms=self._move_ms,
            dry_run=self._dry_run,
        )


def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GrabSequence()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
