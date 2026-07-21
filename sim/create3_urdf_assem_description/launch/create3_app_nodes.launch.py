#!/usr/bin/env python3
"""Create 3 application nodes without ros2_control spawners.

Same as irobot_create_common_bringup/create3_nodes.launch.py but omits the
controller spawners, which must start only after the Gazebo model is loaded.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('gazebo', default_value='classic',
                          choices=['classic', 'ignition'],
                          description='Which gazebo simulator to use'),
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
]


def generate_launch_description():
    pkg = get_package_share_directory('irobot_create_common_bringup')

    def cfg(name):
        return PathJoinSubstitution([pkg, 'config', name])

    nodes = [
        Node(package='irobot_create_nodes',
             name='hazards_vector_publisher',
             executable='hazards_vector_publisher',
             parameters=[cfg('hazard_vector_params.yaml'), {'use_sim_time': True}],
             output='screen'),
        Node(package='irobot_create_nodes',
             name='ir_intensity_vector_publisher',
             executable='ir_intensity_vector_publisher',
             parameters=[cfg('ir_intensity_vector_params.yaml'), {'use_sim_time': True}],
             output='screen'),
        Node(package='irobot_create_nodes',
             name='motion_control',
             executable='motion_control',
             parameters=[{'use_sim_time': True, 'safety_override': 'backup_only'}],
             output='screen',
             remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]),
        Node(package='irobot_create_nodes',
             name='wheel_status_publisher',
             executable='wheel_status_publisher',
             parameters=[cfg('wheel_status_params.yaml'), {'use_sim_time': True}],
             output='screen'),
        Node(package='irobot_create_nodes',
             name='mock_publisher',
             executable='mock_publisher',
             parameters=[cfg('mock_params.yaml'), {'use_sim_time': True}],
             output='screen'),
        Node(package='irobot_create_nodes',
             name='robot_state',
             executable='robot_state',
             parameters=[cfg('robot_state_params.yaml'), {'use_sim_time': True}],
             output='screen'),
        Node(package='irobot_create_nodes',
             name='kidnap_estimator_publisher',
             executable='kidnap_estimator_publisher',
             parameters=[cfg('kidnap_estimator_params.yaml'), {'use_sim_time': True}],
             output='screen'),
        Node(package='irobot_create_nodes',
             name='ui_mgr',
             executable='ui_mgr',
             parameters=[cfg('ui_mgr_params.yaml'),
                         {'use_sim_time': True},
                         {'gazebo': LaunchConfiguration('gazebo')}],
             output='screen'),
    ]

    ld = LaunchDescription(ARGUMENTS)
    for node in nodes:
        ld.add_action(node)
    return ld
