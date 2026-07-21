#!/usr/bin/env python3
"""Bring up the Create 3 + arm in Gazebo Harmonic.

Reuses iRobot's create3_sim world, ros_gz bridges and Create 3 nodes, but spawns
OUR combined description (create3_with_arm.urdf.xacro) instead of the stock
Create 3, and adds the arm controllers plus the sim <-> arm interface bridge.

    ros2 launch create3_urdf_assem_description create3_arm_gz.launch.py

Common overrides:
    world:=maze  use_rviz:=false  spawn_dock:=false
    arm_mount_z:=0.038             # tune where the arm sits on the base
"""

from ament_index_python.packages import get_package_share_directory

from irobot_create_common_bringup.namespace import GetNamespacedName
from irobot_create_common_bringup.offset import (
    OffsetParser, RotationalOffsetX, RotationalOffsetY)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction,
    IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable,
    TimerAction, UnsetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue


ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('world', default_value='depot',
                          description='Gazebo world (depot or maze)'),
    DeclareLaunchArgument('use_rviz', default_value='true',
                          choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('spawn_dock', default_value='true',
                          choices=['true', 'false'],
                          description='Spawn the standard dock model.'),
]

for mount_element in ['arm_mount_x', 'arm_mount_y',
                      'arm_mount_roll', 'arm_mount_pitch', 'arm_mount_yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(
        mount_element, default_value='0.0',
        description=f'{mount_element} component of the arm mount on base_link.'))

ARGUMENTS.append(DeclareLaunchArgument(
    'arm_mount_z', default_value='0.038',
    description='Vertical offset of the arm mount on base_link (meters).'))

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():
    pkg_desc = get_package_share_directory('create3_urdf_assem_description')
    pkg_gz_bringup = get_package_share_directory('irobot_create_gz_bringup')
    pkg_common_bringup = get_package_share_directory('irobot_create_common_bringup')

    sim_launch = PathJoinSubstitution([pkg_desc, 'launch', 'gz_sim.launch.py'])
    bridge_launch = PathJoinSubstitution(
        [pkg_gz_bringup, 'launch', 'create3_ros_gz_bridge.launch.py'])
    gz_nodes_launch = PathJoinSubstitution(
        [pkg_gz_bringup, 'launch', 'create3_gz_nodes.launch.py'])
    app_nodes_launch = PathJoinSubstitution(
        [pkg_desc, 'launch', 'create3_app_nodes.launch.py'])
    controllers_launch = PathJoinSubstitution(
        [pkg_desc, 'launch', 'create3_controllers.launch.py'])
    dock_description_launch = PathJoinSubstitution(
        [pkg_desc, 'launch', 'dock_description_gz.launch.py'])
    rviz_launch = PathJoinSubstitution(
        [pkg_common_bringup, 'launch', 'rviz2.launch.py'])

    xacro_file = PathJoinSubstitution(
        [pkg_desc, 'urdf', 'create3_with_arm.urdf.xacro'])

    namespace = LaunchConfiguration('namespace')
    world = LaunchConfiguration('world')
    x, y, z = (LaunchConfiguration('x'), LaunchConfiguration('y'),
               LaunchConfiguration('z'))
    yaw = LaunchConfiguration('yaw')

    robot_name = GetNamespacedName(namespace, 'create3')
    dock_name = GetNamespacedName(namespace, 'standard_dock')

    # Dock spawn offset (same as iRobot create3_spawn.launch.py).
    dock_offset_x = RotationalOffsetX(0.157, yaw)
    dock_offset_y = RotationalOffsetY(0.157, yaw)
    x_dock = OffsetParser(x, dock_offset_x)
    y_dock = OffsetParser(y, dock_offset_y)
    yaw_dock = OffsetParser(yaw, 3.1416)

    robot_description_command = Command([
        'xacro', ' ', xacro_file,
        ' gazebo:=ignition',
        ' namespace:=', namespace,
        ' control_yaml:=', PathJoinSubstitution(
            [pkg_desc, 'config', 'sim_controllers.yaml']),
        ' arm_mount_x:=', LaunchConfiguration('arm_mount_x'),
        ' arm_mount_y:=', LaunchConfiguration('arm_mount_y'),
        ' arm_mount_z:=', LaunchConfiguration('arm_mount_z'),
        ' arm_mount_roll:=', LaunchConfiguration('arm_mount_roll'),
        ' arm_mount_pitch:=', LaunchConfiguration('arm_mount_pitch'),
        ' arm_mount_yaw:=', LaunchConfiguration('arm_mount_yaw'),
    ])

    dock_urdf_file = PathJoinSubstitution(
        [pkg_desc, 'urdf', 'standard_dock_gz.urdf'])

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-world', world,
                   '-name', robot_name,
                   '-x', x, '-y', y, '-z', z, '-Y', yaw,
                   '-string', robot_description_command],
        output='screen',
    )

    robot_description_broadcaster = Node(
        package='robot_arm',
        executable='robot_description_broadcaster',
        name='robot_description_broadcaster',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': ParameterValue(
                robot_description_command, value_type=str)},
        ],
    )

    # Publishes zero joint states until joint_state_broadcaster takes over.
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': ParameterValue(
                robot_description_command, value_type=str)},
        ],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    spawn_dock = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-world', world,
                   '-name', dock_name,
                   '-x', x_dock, '-y', y_dock, '-z', z, '-Y', yaw_dock,
                   '-file', dock_urdf_file],
        output='screen',
        condition=IfCondition(LaunchConfiguration('spawn_dock')),
    )

    # Spawn as soon as the description nodes are up (meshes use file:// paths).
    delayed_robot_spawn = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[
                TimerAction(period=1.0, actions=[spawn_robot]),
            ],
        )
    )

    # Controllers must start only after Gazebo finishes loading the model.
    delayed_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                TimerAction(period=15.0, actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource([controllers_launch]),
                    ),
                ]),
            ],
        )
    )

    delayed_dock_spawn = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[
                # Give dock_state_publisher time to latch standard_dock_description.
                TimerAction(period=3.0, actions=[spawn_dock]),
            ],
        )
    )

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([sim_launch]),
        launch_arguments=[('world', world)])

    spawn_group = GroupAction([
        PushRosNamespace(namespace),

        robot_description_broadcaster,
        joint_state_publisher,
        robot_state_publisher,

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([dock_description_launch]),
            condition=IfCondition(LaunchConfiguration('spawn_dock')),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([bridge_launch]),
            launch_arguments=[('world', world),
                              ('robot_name', robot_name),
                              ('dock_name', dock_name)]),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([app_nodes_launch]),
            launch_arguments=[('gazebo', 'ignition'), ('namespace', namespace)]),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([gz_nodes_launch]),
            launch_arguments=[('robot_name', robot_name),
                              ('dock_name', dock_name)]),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([rviz_launch]),
            condition=IfCondition(LaunchConfiguration('use_rviz'))),
    ])

    local_dds = [
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_SUPER_CLIENT'),
        SetEnvironmentVariable(
            name='ROS_AUTOMATIC_DISCOVERY_RANGE',
            value='SUBNET'),
    ]

    ld = LaunchDescription(ARGUMENTS)
    for action in local_dds:
        ld.add_action(action)
    ld.add_action(sim)
    ld.add_action(spawn_group)
    ld.add_action(delayed_robot_spawn)
    ld.add_action(delayed_dock_spawn)
    ld.add_action(delayed_controllers)
    return ld
