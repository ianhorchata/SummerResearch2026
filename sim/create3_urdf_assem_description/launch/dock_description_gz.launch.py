#!/usr/bin/env python3
"""Bring up the standard Create 3 dock in Gazebo Harmonic.

Publishes the dock URDF/TF and spawns the model in Gazebo (file:// mesh paths).

    ros2 launch create3_urdf_assem_description dock_description_gz.launch.py
"""

from ament_index_python.packages import get_package_share_directory

from irobot_create_common_bringup.namespace import GetNamespacedName

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('world', default_value='depot',
                          description='Gazebo world'),
    DeclareLaunchArgument('visualize_rays', default_value='false',
                          choices=['true', 'false'],
                          description='Enable/disable IR ray visualization'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the dock pose.'))


def generate_launch_description():
    pkg_desc = get_package_share_directory('create3_urdf_assem_description')

    sim_launch = PathJoinSubstitution([pkg_desc, 'launch', 'gz_sim.launch.py'])
    dock_xacro = PathJoinSubstitution([pkg_desc, 'urdf', 'standard_dock_gz.urdf.xacro'])
    dock_urdf_file = PathJoinSubstitution([pkg_desc, 'urdf', 'standard_dock_gz.urdf'])

    namespace = LaunchConfiguration('namespace')
    world = LaunchConfiguration('world')
    x, y, z, yaw = (LaunchConfiguration('x'), LaunchConfiguration('y'),
                    LaunchConfiguration('z'), LaunchConfiguration('yaw'))
    dock_name = GetNamespacedName(namespace, 'standard_dock')

    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='dock_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': ParameterValue(
                Command(['xacro', ' ', dock_xacro, ' gazebo:=ignition']),
                value_type=str)},
        ],
        remappings=[
            ('robot_description', 'standard_dock_description'),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )

    tf_odom_std_dock_link_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_odom_std_dock_link_publisher',
        arguments=['0.157', '0', '0', '3.141592', '0', '0', 'odom', 'std_dock_link'],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        output='screen',
    )

    spawn_dock = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-world', world,
                   '-name', dock_name,
                   '-x', x, '-y', y, '-z', z, '-Y', yaw,
                   '-file', dock_urdf_file],
        output='screen',
    )

    # Give Gazebo time to start before spawning the dock model.
    delayed_dock_spawn = RegisterEventHandler(
        OnProcessStart(
            target_action=state_publisher,
            on_start=[
                TimerAction(period=3.0, actions=[spawn_dock]),
            ],
        )
    )

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([sim_launch]),
        launch_arguments=[('world', world)])

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(sim)
    ld.add_action(state_publisher)
    ld.add_action(tf_odom_std_dock_link_publisher)
    ld.add_action(delayed_dock_spawn)
    return ld
