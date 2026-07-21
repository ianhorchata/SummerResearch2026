#!/usr/bin/env python3
"""Publish /robot_description with transient_local durability.

robot_state_publisher latches the URDF once at startup; some subscribers
(including RViz in certain timing windows) can miss it. This node republishes
the same URDF string so RViz and other late joiners can latch it reliably.
"""

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
import rclpy
from std_msgs.msg import String


class RobotDescriptionBroadcaster(Node):
    def __init__(self) -> None:
        super().__init__('robot_description_broadcaster')
        self.declare_parameter('robot_description', '')
        urdf = self.get_parameter('robot_description').get_parameter_value().string_value
        if not urdf:
            self.get_logger().fatal('robot_description parameter is empty')
            raise RuntimeError('robot_description parameter is empty')

        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(String, '/robot_description', qos)
        self._msg = String(data=urdf)
        # Publish several times at startup so gz_ros2_control / RViz can latch.
        for _ in range(5):
            self._pub.publish(self._msg)
        self.create_timer(1.0, self._republish)
        self.get_logger().info(f'publishing robot_description ({len(urdf)} bytes)')

    def _republish(self) -> None:
        self._pub.publish(self._msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotDescriptionBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
