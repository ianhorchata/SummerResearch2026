#!/usr/bin/env python3
"""Gazebo Harmonic launcher with Create 3 + arm mesh resource paths.

Same as irobot_create_gz_bringup/sim.launch.py, but GZ_SIM_RESOURCE_PATH also
includes create3_urdf_assem_description so model:// URIs resolve at spawn time.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'],
                          description='use_sim_time'),
    DeclareLaunchArgument('world', default_value='depot',
                          description='Ignition World'),
    DeclareLaunchArgument(
        'physics_engine',
        default_value='gz-physics-bullet-featherstone-plugin',
        description='Physics engine (bullet-featherstone required for gripper mimics)'),
]


def generate_launch_description():
    pkg_irobot_create_gz_bringup = get_package_share_directory(
        'irobot_create_gz_bringup')
    pkg_irobot_create_gz_plugins = get_package_share_directory(
        'irobot_create_gz_plugins')
    pkg_irobot_create_description = get_package_share_directory(
        'irobot_create_description')
    pkg_arm_description = get_package_share_directory(
        'create3_urdf_assem_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join([
            os.path.join(pkg_irobot_create_gz_bringup, 'worlds'),
            str(Path(pkg_irobot_create_description).parent.resolve()),
            str(Path(pkg_arm_description).parent.resolve()),
        ]))

    gz_gui_plugin_path = SetEnvironmentVariable(
        name='GZ_GUI_PLUGIN_PATH',
        value=os.path.join(pkg_irobot_create_gz_plugins, 'lib'))

    gz_sim_launch = PathJoinSubstitution(
        [pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gz_sim_launch]),
        launch_arguments=[(
            'gz_args', [
                LaunchConfiguration('world'),
                '.sdf',
                ' -r -v 4',
                ' --physics-engine ', LaunchConfiguration('physics_engine'),
                ' --gui-config ',
                PathJoinSubstitution(
                    [pkg_irobot_create_gz_bringup, 'gui', 'create3', 'gui.config']
                )
            ]
        )]
    )

    clock_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='clock_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ])

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gz_resource_path)
    ld.add_action(gz_gui_plugin_path)
    ld.add_action(gz_sim)
    ld.add_action(clock_bridge)
    return ld
