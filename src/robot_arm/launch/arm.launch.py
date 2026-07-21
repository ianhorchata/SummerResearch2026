"""Launch just the arm node. Override params on the command line, e.g.

    ros2 launch robot_arm arm.launch.py port:=/dev/ttyACM1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port = LaunchConfiguration("port")
    scan_max_id = LaunchConfiguration("scan_max_id")

    # Map physical servo ids to the URDF joint names so robot_state_publisher /
    # RViz can match them. Format: "<id>:<joint name>[:<center_units>[:<sign>]]".
    # center_units defaults to 500 (servo midpoint), sign defaults to +1.
    # Flip the sign to -1 if a joint rotates the wrong way in RViz; tweak the
    # center so the arm's resting pose lines up with the URDF zero pose.
    # center = servo units at the upright/neutral pose (gripper closed), so that
    # pose maps to the URDF zero.
    default_joint_name_map = [
        "1:Servo1:481:-1",
        "2:Servo2:488:-1",
        "3:Servo3:531:-1",
        "4:Servo4:900:1",
    ]

    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("scan_max_id", default_value="20"),
        Node(
            package="robot_arm",
            executable="arm_node",
            name="arm_node",
            output="screen",
            parameters=[{
                "port": port,
                "scan_max_id": scan_max_id,
                "joint_name_map": default_joint_name_map,
            }],
        ),
    ])
