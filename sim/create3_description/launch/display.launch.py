#!/usr/bin/env python3
"""Display the robot in RViz (no hardware required).

    ros2 launch create3_description display.launch.py
    ros2 launch create3_description display.launch.py use_gui:=false   # live arm

Stereo camera TFs match robot_bringup/robot.launch.py (CAD inches -> meters,
10 deg yaw toe-in). Keep both in sync if the mount pose changes.
"""

import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

_IN2M = 0.0254
_CAM_X = 1.467 * _IN2M
_CAM_Y = 5.185 * _IN2M
_CAM_Z = 4.0 * _IN2M
_TOE_IN = math.radians(10.0)
_OPTICAL_RPY = (-math.pi / 2.0, 0.0, -math.pi / 2.0)


def _static_tf(x, y, z, roll, pitch, yaw, parent, child):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"static_tf_{child}",
        arguments=[
            "--x", str(x), "--y", str(y), "--z", str(z),
            "--roll", str(roll), "--pitch", str(pitch), "--yaw", str(yaw),
            "--frame-id", parent,
            "--child-frame-id", child,
        ],
    )


def _camera_tfs():
    ox, oy, oz = _OPTICAL_RPY
    return [
        _static_tf(_CAM_X, _CAM_Y, _CAM_Z, 0.0, 0.0, -_TOE_IN,
                   "base_link", "left_camera_link"),
        _static_tf(_CAM_X, -_CAM_Y, _CAM_Z, 0.0, 0.0, _TOE_IN,
                   "base_link", "right_camera_link"),
        _static_tf(0.0, 0.0, 0.0, ox, oy, oz,
                   "left_camera_link", "left_camera_optical_frame"),
        _static_tf(0.0, 0.0, 0.0, ox, oy, oz,
                   "right_camera_link", "right_camera_optical_frame"),
    ]


def generate_launch_description():
    pkg = FindPackageShare("create3_description")
    use_gui = LaunchConfiguration("use_gui")
    rviz_config = PathJoinSubstitution([pkg, "config", "display.rviz"])

    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution([pkg, "urdf", "create3.xacro"]),
    ])
    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content, value_type=str)
    }

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_gui", default_value="true",
            description="true = slider GUI; false = live /arm/joint_states"),

        # GUI mode: sliders -> /joint_states -> TF
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
            condition=IfCondition(use_gui),
        ),

        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            condition=IfCondition(use_gui),
        ),

        # Live mode: wire arm_node straight into robot_state_publisher so joint
        # names only need to match once (Servo1..Servo4 in URDF and arm.launch).
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
            remappings=[("joint_states", "/arm/joint_states")],
            condition=UnlessCondition(use_gui),
        ),

        *(_camera_tfs()),

        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ])
