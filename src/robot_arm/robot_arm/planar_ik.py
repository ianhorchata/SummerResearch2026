"""Planar 3-DOF inverse / forward kinematics for the Create 3 arm.

Geometry is taken from ``sim/create3_urdf_assem_description/urdf/arm.urdf.xacro``
(and the default mount in ``create3_with_arm.urdf.xacro``). All three arm
joints rotate about local Y, so the workspace is the sagittal (X–Z) plane.

Joint angles are URDF / ``/arm/joint_states`` radians (same convention as
``DEFAULT_GRAB`` in ``grab_sequence.py``). Gripper is not part of the IK.

Typical use::

    from robot_arm.planar_ik import PlanarArmKinematics, solve_grab_joints

    kin = PlanarArmKinematics()
    q = kin.ik(x=0.38, z=0.06)          # tip in base_link
    # or:
    q = solve_grab_joints(0.38, 0.06)   # includes gripper-open default
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# URDF constants (metres / radians)
# ---------------------------------------------------------------------------

# base_link -> arm_mount (create3_with_arm default)
ARM_MOUNT_XYZ = (0.0, 0.0, 0.038)
# arm_mount -> ArmAssem tower
ARM_TOWER_XYZ = (0.084, 0.0, 0.0936)
# tower -> joint 1 origin
JOINT1_ORIGIN = (0.018763, 0.0265, 0.0269)
# consecutive link lengths along +Z of the parent joint frame
LINK1_LENGTH = 0.128317  # joint1 -> joint2
LINK2_LENGTH = 0.128317  # joint2 -> joint3
# joint3 child -> gripper Base_v8_1
GRIPPER_BASE_XYZ = (0.0, -0.02715, 0.037917)

JOINT_LIMIT = 2.094395  # ±120 deg from URDF

# Grasp-center offset in the gripper Base frame (tune if fingers miss).
# Chosen so FK(DEFAULT_GRAB) lands near the taught 15" standoff height.
DEFAULT_EE_TIP_XYZ = (0.042, 0.0, 0.055)

# Taught grab joints (Servo1–3); pitch = sum for RotY chain.
_DEFAULT_SEED = (1.32, 1.0890854532444616, -0.9131562646434331)
DEFAULT_GRAB_PITCH = sum(_DEFAULT_SEED)

# Gripper-open used when packing a 4-vector grab pose.
_DEFAULT_GRIPPER_OPEN = -1.0807078728348887


def _rot_y(theta: float) -> Tuple[Tuple[float, float, float], ...]:
    c, s = math.cos(theta), math.sin(theta)
    return (
        (c, 0.0, s),
        (0.0, 1.0, 0.0),
        (-s, 0.0, c),
    )


def _mat_vec(
    r: Tuple[Tuple[float, float, float], ...], v: Sequence[float]
) -> Tuple[float, float, float]:
    return (
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    )


def _add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@dataclass(frozen=True)
class IkSolution:
    """Joint angles (rad) for Servo1–3 and the tip pose they realize."""

    joints: Tuple[float, float, float]
    tip_x: float
    tip_y: float
    tip_z: float
    pitch: float


class PlanarArmKinematics:
    """Analytical planar FK/IK for arm_joint_1/2/3."""

    def __init__(
        self,
        *,
        mount_xyz: Sequence[float] = ARM_MOUNT_XYZ,
        tower_xyz: Sequence[float] = ARM_TOWER_XYZ,
        joint1_origin: Sequence[float] = JOINT1_ORIGIN,
        link1: float = LINK1_LENGTH,
        link2: float = LINK2_LENGTH,
        gripper_base_xyz: Sequence[float] = GRIPPER_BASE_XYZ,
        ee_tip_xyz: Sequence[float] = DEFAULT_EE_TIP_XYZ,
        joint_limit: float = JOINT_LIMIT,
    ) -> None:
        self.mount_xyz = tuple(float(x) for x in mount_xyz)
        self.tower_xyz = tuple(float(x) for x in tower_xyz)
        self.joint1_origin = tuple(float(x) for x in joint1_origin)
        self.link1 = float(link1)
        self.link2 = float(link2)
        self.gripper_base_xyz = tuple(float(x) for x in gripper_base_xyz)
        self.ee_tip_xyz = tuple(float(x) for x in ee_tip_xyz)
        self.joint_limit = float(joint_limit)

        self.shoulder = _add(_add(self.mount_xyz, self.tower_xyz), self.joint1_origin)
        # Offset from joint-3 frame origin to tip, expressed in the last link frame.
        self.tool_offset = _add(self.gripper_base_xyz, self.ee_tip_xyz)

    def fk(self, q: Sequence[float]) -> IkSolution:
        """Forward kinematics: joints (rad) -> tip pose in base_link."""
        if len(q) < 3:
            raise ValueError(f"expected 3 joint angles, got {len(q)}")
        th1, th2, th3 = float(q[0]), float(q[1]), float(q[2])
        j2 = _add(self.shoulder, _mat_vec(_rot_y(th1), (0.0, 5e-5, self.link1)))
        j3 = _add(j2, _mat_vec(_rot_y(th1 + th2), (0.0, 5e-5, self.link2)))
        pitch = th1 + th2 + th3
        tip = _add(j3, _mat_vec(_rot_y(pitch), self.tool_offset))
        return IkSolution(
            joints=(th1, th2, th3),
            tip_x=tip[0],
            tip_y=tip[1],
            tip_z=tip[2],
            pitch=pitch,
        )

    def ik(
        self,
        x: float,
        z: float,
        *,
        pitch: float = DEFAULT_GRAB_PITCH,
        seed: Sequence[float] = _DEFAULT_SEED,
        y: Optional[float] = None,
    ) -> IkSolution:
        """Solve for tip (x, z) in base_link at the given wrist pitch.

        ``y`` is accepted for API completeness; planar IK ignores it (the arm
        has no yaw). Raises ``ValueError`` if unreachable or all solutions
        violate joint limits.
        """
        del y  # planar: lateral offset must be handled by base drive
        tip = (float(x), 0.0, float(z))
        wrist = _sub(tip, _mat_vec(_rot_y(pitch), self.tool_offset))

        dx = wrist[0] - self.shoulder[0]
        dz = wrist[2] - self.shoulder[2]
        r2 = dx * dx + dz * dz
        reach = self.link1 + self.link2
        if r2 < 1e-12:
            raise ValueError("IK target coincides with shoulder")
        if r2 > reach * reach * (1.0 + 1e-9):
            raise ValueError(
                f"IK unreachable: wrist planar range {math.sqrt(r2):.3f}m "
                f"> {reach:.3f}m"
            )

        cos_t2 = (r2 - self.link1**2 - self.link2**2) / (2.0 * self.link1 * self.link2)
        cos_t2 = max(-1.0, min(1.0, cos_t2))
        t2_mag = math.acos(cos_t2)

        candidates: list[Tuple[float, float, float]] = []
        for t2 in (t2_mag, -t2_mag):
            t1 = math.atan2(dx, dz) - math.atan2(
                self.link2 * math.sin(t2),
                self.link1 + self.link2 * math.cos(t2),
            )
            t3 = float(pitch) - t1 - t2
            candidates.append((t1, t2, t3))

        lim = self.joint_limit
        valid = [
            c for c in candidates
            if all(abs(a) <= lim + 1e-9 for a in c)
        ]
        if not valid:
            raise ValueError(
                "IK solutions exceed joint limits "
                f"(±{math.degrees(lim):.0f} deg): {candidates}"
            )

        seed_t = tuple(float(s) for s in seed[:3])

        def _dist(c: Tuple[float, float, float]) -> float:
            return sum((a - b) ** 2 for a, b in zip(c, seed_t))

        best = min(valid, key=_dist)
        return self.fk(best)

    def within_reach(self, x: float, z: float, pitch: float = DEFAULT_GRAB_PITCH) -> bool:
        try:
            self.ik(x, z, pitch=pitch)
            return True
        except ValueError:
            return False


def solve_grab_joints(
    x: float,
    z: float,
    *,
    pitch: float = DEFAULT_GRAB_PITCH,
    gripper_open_rad: float = _DEFAULT_GRIPPER_OPEN,
    ee_tip_xyz: Sequence[float] = DEFAULT_EE_TIP_XYZ,
    seed: Sequence[float] = _DEFAULT_SEED,
) -> Tuple[float, float, float, float]:
    """Return a 4-vector grab pose [j1, j2, j3, gripper] for ``run_grab_sequence``."""
    kin = PlanarArmKinematics(ee_tip_xyz=ee_tip_xyz)
    sol = kin.ik(x, z, pitch=pitch, seed=seed)
    return (*sol.joints, float(gripper_open_rad))
