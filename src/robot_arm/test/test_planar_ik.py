"""Unit tests for planar arm IK (no ROS runtime required)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# Allow `python3 test_planar_ik.py` without installing the package.
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from robot_arm.planar_ik import (  # noqa: E402
    DEFAULT_GRAB_PITCH,
    PlanarArmKinematics,
    solve_grab_joints,
)

# Taught grab seed (matches grab_sequence.DEFAULT_GRAB[:3]).
_DEFAULT_GRAB = (1.32, 1.0890854532444616, -0.9131562646434331)


class TestPlanarIk(unittest.TestCase):
    def setUp(self) -> None:
        self.kin = PlanarArmKinematics()

    def test_fk_default_grab_is_forward_and_low(self) -> None:
        sol = self.kin.fk(_DEFAULT_GRAB)
        self.assertAlmostEqual(sol.pitch, DEFAULT_GRAB_PITCH, places=9)
        self.assertGreater(sol.tip_x, 0.3)
        self.assertLess(sol.tip_z, 0.15)
        self.assertAlmostEqual(sol.tip_y, 0.0, places=2)

    def test_ik_roundtrip_default_grab(self) -> None:
        fk = self.kin.fk(_DEFAULT_GRAB)
        ik = self.kin.ik(fk.tip_x, fk.tip_z, pitch=fk.pitch, seed=_DEFAULT_GRAB)
        for a, b in zip(ik.joints, _DEFAULT_GRAB):
            self.assertAlmostEqual(a, b, places=9)

    def test_ik_standoff_target(self) -> None:
        sol = self.kin.ik(0.38, 0.06, pitch=DEFAULT_GRAB_PITCH, seed=_DEFAULT_GRAB)
        self.assertTrue(all(abs(q) <= self.kin.joint_limit for q in sol.joints))
        self.assertAlmostEqual(sol.tip_x, 0.38, places=6)
        self.assertAlmostEqual(sol.tip_z, 0.06, places=6)

    def test_unreachable_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.kin.ik(2.0, 0.06)

    def test_solve_grab_joints_length(self) -> None:
        q = solve_grab_joints(0.38, 0.06)
        self.assertEqual(len(q), 4)
        self.assertTrue(math.isfinite(q[3]))


if __name__ == "__main__":
    unittest.main()
