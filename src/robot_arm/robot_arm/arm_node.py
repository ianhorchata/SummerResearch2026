#!/usr/bin/env python3
"""ROS 2 node wrapping the Hiwonder LX serial-bus servo arm.

Design: the serial protocol lives entirely in ``hiwonder_servo.py`` (unchanged).
This node is a thin adapter that exposes the bus over ROS 2:

Topics
  publishes  arm/joint_states   sensor_msgs/JointState   (positions in RADIANS)
  publishes  arm/health         robot_interfaces/ServoHealth
  subscribes arm/joint_command  sensor_msgs/JointState   (positions in RADIANS)

Services
  arm/scan        std_srvs/Trigger              re-scan the bus for servo ids
  arm/home        std_srvs/Trigger              move every servo to center (500)
  arm/set_torque  std_srvs/SetBool              enable/disable holding torque
  arm/move_joints robot_interfaces/MoveJoints   move ids to angles (DEGREES)

The bus is half-duplex and every call blocks on serial I/O, so this node uses
the default single-threaded executor: callbacks never run concurrently and can
safely share one HiwonderServoBus instance.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, SetBool

from robot_interfaces.msg import ServoHealth
from robot_interfaces.srv import MoveJoints

from .hiwonder_servo import HiwonderServoBus, ServoError, DEG_PER_UNIT, POS_MIN, POS_MAX

# 0..1000 servo units span 240 degrees; convert to/from radians for ROS.
_RAD_PER_UNIT = math.radians(DEG_PER_UNIT)


def _default_servo_name(servo_id: int) -> str:
    return f"servo_{servo_id}"


class ArmNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_node")

        # --- parameters (override with --ros-args -p port:=/dev/ttyACM1 etc.) ---
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("scan_max_id", 20)
        self.declare_parameter("state_rate_hz", 10.0)
        self.declare_parameter("health_rate_hz", 1.0)
        self.declare_parameter("default_move_ms", 800)
        # Map servo ids to the joint names used in the URDF so RViz /
        # robot_state_publisher can match them, with optional calibration.
        #
        #   format: "<id>:<joint name>[:<center_units>[:<sign>]]"
        #
        # The published angle is:  (units - center_units) * rad_per_unit * sign
        # center_units defaults to 500 (servo midpoint), sign defaults to +1.
        # Flip sign to -1 if the model rotates the wrong way; adjust center so
        # the servo's resting pose matches the URDF's zero pose.
        #
        #   e.g. ["1:Revolute 14:500:-1", "2:Revolute 7:480:1"]
        #
        # Unmapped ids fall back to name "servo_<id>" with center 0, sign +1.
        self.declare_parameter("joint_name_map", [""])

        port = self.get_parameter("port").value
        baud = int(self.get_parameter("baud").value)
        self.scan_max_id = int(self.get_parameter("scan_max_id").value)
        self.default_move_ms = int(self.get_parameter("default_move_ms").value)

        self._build_joint_name_map()

        try:
            self.bus = HiwonderServoBus(port, baud)
        except ServoError as exc:
            self.get_logger().fatal(f"could not open servo bus on {port}: {exc}")
            raise

        # --- discover which servos are physically on the bus ------------------
        self.servo_ids = self._scan()
        if self.servo_ids:
            self.get_logger().info(f"discovered servo ids: {self.servo_ids}")
        else:
            self.get_logger().warn(
                "no servos responded on the bus "
                "(check power, wiring, baud, and scan_max_id)")

        # --- publishers -------------------------------------------------------
        self.state_pub = self.create_publisher(JointState, "arm/joint_states", 10)
        self.health_pub = self.create_publisher(ServoHealth, "arm/health", 10)

        # --- subscriber -------------------------------------------------------
        self.create_subscription(
            JointState, "arm/joint_command", self._on_command, 10)

        # --- services ---------------------------------------------------------
        self.create_service(Trigger, "arm/scan", self._on_scan)
        self.create_service(Trigger, "arm/home", self._on_home)
        self.create_service(SetBool, "arm/set_torque", self._on_set_torque)
        self.create_service(MoveJoints, "arm/move_joints", self._on_move_joints)

        # --- periodic timers --------------------------------------------------
        state_hz = float(self.get_parameter("state_rate_hz").value)
        health_hz = float(self.get_parameter("health_rate_hz").value)
        if state_hz > 0:
            self.create_timer(1.0 / state_hz, self._publish_state)
        if health_hz > 0:
            self.create_timer(1.0 / health_hz, self._publish_health)

        self.get_logger().info("arm_node ready")

    # ----------------------------------------------------------------- helpers
    def _build_joint_name_map(self) -> None:
        """Parse the joint_name_map parameter into id<->name+calibration lookups."""
        self._id_to_name = {}
        self._name_to_id = {}
        self._calib = {}  # servo_id -> (center_units, sign)
        for entry in self.get_parameter("joint_name_map").value or []:
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) < 2 or not parts[1].strip():
                self.get_logger().warn(f"ignoring bad joint_name_map entry {entry!r}")
                continue
            try:
                sid = int(parts[0])
                name = parts[1].strip()
                center = int(parts[2]) if len(parts) > 2 and parts[2] else 500
                sign = int(parts[3]) if len(parts) > 3 and parts[3] else 1
            except ValueError:
                self.get_logger().warn(f"ignoring bad joint_name_map entry {entry!r}")
                continue
            self._id_to_name[sid] = name
            self._name_to_id[name] = sid
            self._calib[sid] = (center, 1 if sign >= 0 else -1)

    def _joint_name(self, servo_id: int) -> str:
        return self._id_to_name.get(servo_id, _default_servo_name(servo_id))

    def _servo_id(self, joint_name: str):
        """Resolve a joint name back to a servo id, or None if unknown."""
        if joint_name in self._name_to_id:
            return self._name_to_id[joint_name]
        # fall back to the default "servo_<id>" convention
        try:
            return int(joint_name.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return None

    def _units_to_rad(self, servo_id: int, units: int) -> float:
        center, sign = self._calib.get(servo_id, (0, 1))
        return (units - center) * _RAD_PER_UNIT * sign

    def _rad_to_units(self, servo_id: int, rad: float) -> int:
        center, sign = self._calib.get(servo_id, (0, 1))
        return int(round(center + (rad / (_RAD_PER_UNIT * sign))))

    def _scan(self):
        try:
            return self.bus.scan(range(0, self.scan_max_id + 1))
        except ServoError as exc:
            self.get_logger().error(f"bus scan failed: {exc}")
            return []

    # -------------------------------------------------------------- publishers
    def _publish_state(self) -> None:
        if not self.servo_ids:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for sid in self.servo_ids:
            try:
                pos_units = self.bus.read_position(sid)
            except ServoError as exc:
                self.get_logger().warn(f"read_position({sid}) failed: {exc}")
                continue
            msg.name.append(self._joint_name(sid))
            msg.position.append(self._units_to_rad(sid, pos_units))
        if msg.name:
            self.state_pub.publish(msg)

    def _publish_health(self) -> None:
        if not self.servo_ids:
            return
        msg = ServoHealth()
        msg.header.stamp = self.get_clock().now().to_msg()
        for sid in self.servo_ids:
            try:
                msg.voltage.append(self.bus.read_voltage(sid))
                msg.temperature.append(self.bus.read_temperature(sid))
                msg.torque_on.append(self.bus.is_torque_on(sid))
                msg.issues.append("; ".join(self.bus.health(sid)))
                msg.ids.append(sid)
            except ServoError as exc:
                self.get_logger().warn(f"health({sid}) failed: {exc}")
        self.health_pub.publish(msg)

    # ------------------------------------------------------------- subscribers
    def _on_command(self, msg: JointState) -> None:
        """Move servos named in the JointState (positions interpreted as radians)."""
        for name, rad in zip(msg.name, msg.position):
            sid = self._servo_id(name)
            if sid is None:
                self.get_logger().warn(f"ignoring unknown joint name {name!r}")
                continue
            units = self._rad_to_units(sid, rad)
            units = max(POS_MIN, min(POS_MAX, units))
            try:
                self.bus.move(sid, units, self.default_move_ms)
            except ServoError as exc:
                self.get_logger().warn(f"move({sid}) failed: {exc}")

    # ---------------------------------------------------------------- services
    def _on_scan(self, request, response):
        self.servo_ids = self._scan()
        response.success = bool(self.servo_ids)
        response.message = f"found ids: {self.servo_ids}"
        return response

    def _on_home(self, request, response):
        if not self.servo_ids:
            response.success = False
            response.message = "no servos on bus"
            return response
        try:
            self.bus.move_many({sid: 500 for sid in self.servo_ids}, time_ms=1500)
            response.success = True
            response.message = f"homed {self.servo_ids}"
        except ServoError as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _on_set_torque(self, request, response):
        ok = True
        for sid in self.servo_ids:
            try:
                self.bus.set_torque(sid, request.data)
            except ServoError as exc:
                ok = False
                self.get_logger().warn(f"set_torque({sid}) failed: {exc}")
        response.success = ok
        response.message = ("torque on" if request.data else "torque off") + \
            f" for {self.servo_ids}"
        return response

    def _on_move_joints(self, request, response):
        if len(request.ids) != len(request.positions_deg):
            response.success = False
            response.message = "ids and positions_deg must be the same length"
            return response
        time_ms = request.time_ms if request.time_ms > 0 else self.default_move_ms
        targets = {}
        for sid, deg in zip(request.ids, request.positions_deg):
            targets[int(sid)] = int(round(deg / DEG_PER_UNIT))
        try:
            self.bus.move_many(targets, time_ms=time_ms)
            response.success = True
            response.message = f"moving {dict(targets)} over {time_ms} ms"
        except ServoError as exc:
            response.success = False
            response.message = str(exc)
        return response

    # ----------------------------------------------------------------- cleanup
    def destroy_node(self) -> bool:
        try:
            self.bus.close()
        except Exception:  # noqa: BLE001  - best-effort cleanup on shutdown
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
