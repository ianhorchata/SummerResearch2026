#!/usr/bin/env python3
"""Simulation stand-in for ``arm_node``.

The real ``arm_node`` talks to the Hiwonder serial servo bus. In Gazebo there is
no serial bus -- the arm is driven by ros2_control (gz_ros2_control) through
``arm_controller`` / ``gripper_controller``. This node keeps the *same* ROS
interface as the real driver so higher-level code is identical in sim and on
hardware:

  subscribes  arm/joint_command   sensor_msgs/JointState   (positions, RADIANS)
  publishes   arm/joint_states    sensor_msgs/JointState   (positions, RADIANS)

Incoming commands are split by joint name into the arm and gripper controllers
and forwarded as single-point JointTrajectory goals. Outgoing states are the arm
subset of the controller_manager's /joint_states, republished under the arm
namespace.

Joint names default to the simulation URDF (arm_joint_1..3, gripper_joint). If
your higher-level code still uses the old "Revolute NN" names, either migrate
those to the new names (recommended for sim2real) or override the *_joints
parameters here.
"""

from rclpy.duration import Duration
from rclpy.node import Node
import rclpy

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class SimArmBridge(Node):
    def __init__(self) -> None:
        super().__init__('sim_arm_bridge')

        self.declare_parameter(
            'arm_joints', ['arm_joint_1', 'arm_joint_2', 'arm_joint_3'])
        self.declare_parameter('gripper_joints', ['gripper_joint'])
        self.declare_parameter('move_time', 0.8)
        self.declare_parameter('arm_trajectory_topic',
                               'arm_controller/joint_trajectory')
        self.declare_parameter('gripper_trajectory_topic',
                               'gripper_controller/joint_trajectory')
        self.declare_parameter('joint_states_topic', 'joint_states')
        self.declare_parameter('command_topic', 'arm/joint_command')
        self.declare_parameter('state_topic', 'arm/joint_states')

        self.arm_joints = list(self.get_parameter('arm_joints').value)
        self.gripper_joints = list(self.get_parameter('gripper_joints').value)
        self.move_time = float(self.get_parameter('move_time').value)
        self._reported = set(self.arm_joints) | set(self.gripper_joints)

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            self.get_parameter('arm_trajectory_topic').value, 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            self.get_parameter('gripper_trajectory_topic').value, 10)
        self.state_pub = self.create_publisher(
            JointState, self.get_parameter('state_topic').value, 10)

        self.create_subscription(
            JointState, self.get_parameter('command_topic').value,
            self._on_command, 10)
        self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self._on_joint_states, 10)

        self.get_logger().info(
            f'sim_arm_bridge ready (arm={self.arm_joints}, '
            f'gripper={self.gripper_joints})')

    def _on_command(self, msg: JointState) -> None:
        arm, gripper = {}, {}
        for name, pos in zip(msg.name, msg.position):
            if name in self.arm_joints:
                arm[name] = pos
            elif name in self.gripper_joints:
                gripper[name] = pos
            else:
                self.get_logger().warn(f'ignoring unknown joint {name!r}')
        if arm:
            self.arm_pub.publish(self._trajectory(arm))
        if gripper:
            self.gripper_pub.publish(self._trajectory(gripper))

    def _trajectory(self, joint_to_pos: dict) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = list(joint_to_pos.keys())
        point = JointTrajectoryPoint()
        point.positions = [float(joint_to_pos[n]) for n in traj.joint_names]
        point.time_from_start = Duration(seconds=self.move_time).to_msg()
        traj.points = [point]
        return traj

    def _on_joint_states(self, msg: JointState) -> None:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        for name, pos in zip(msg.name, msg.position):
            if name in self._reported:
                out.name.append(name)
                out.position.append(pos)
        if out.name:
            self.state_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimArmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
