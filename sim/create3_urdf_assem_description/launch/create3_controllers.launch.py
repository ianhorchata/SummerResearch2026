#!/usr/bin/env python3
"""Spawn Create 3 + arm controllers after the Gazebo model is fully loaded.

Controllers are chained joint_state_broadcaster -> diffdrive_controller ->
arm_controller + gripper_controller so only one switch runs at a time.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_desc = get_package_share_directory('create3_urdf_assem_description')
    pkg_control = get_package_share_directory('irobot_create_control')

    sim_controllers = PathJoinSubstitution(
        [pkg_desc, 'config', 'sim_controllers.yaml'])
    irobot_control = PathJoinSubstitution(
        [pkg_control, 'config', 'control.yaml'])

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'joint_state_broadcaster',
            '-c', 'controller_manager',
            '--param-file', sim_controllers,
            '--controller-manager-timeout', '120',
            '--switch-timeout', '120',
            '--switch-asap',
        ],
    )

    diffdrive_controller = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'diffdrive_controller',
            '-c', 'controller_manager',
            '--param-file', irobot_control,
            '--controller-manager-timeout', '120',
            '--switch-timeout', '120',
            '--switch-asap',
        ],
    )

    arm_controllers = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'arm_controller', 'gripper_controller',
            '-c', 'controller_manager',
            '--param-file', sim_controllers,
            '--controller-manager-timeout', '120',
            '--switch-timeout', '120',
            '--switch-asap',
        ],
    )

    sim_arm_bridge = Node(
        package='robot_arm',
        executable='sim_arm_bridge',
        name='sim_arm_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    diffdrive_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[diffdrive_controller],
        )
    )

    arm_after_diffdrive = RegisterEventHandler(
        OnProcessExit(
            target_action=diffdrive_controller,
            on_exit=[arm_controllers, sim_arm_bridge],
        )
    )

    ld = LaunchDescription()
    ld.add_action(joint_state_broadcaster)
    ld.add_action(diffdrive_after_jsb)
    ld.add_action(arm_after_diffdrive)
    return ld
