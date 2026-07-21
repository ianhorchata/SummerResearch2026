"""Bring up the whole robot: arm node + left/right CSI cameras.

The Create 3 publishes its own ROS 2 topics independently (over the Jetson<->Create
network), so it is NOT launched here -- just `ros2 topic list` to see it.

CSI cameras are published with gscam2 using the Jetson Argus source
(`nvarguscamerasrc`). Camera 1 -> /left, camera 0 -> /right
(sensor-id wiring; swap again in `_spawn_cameras` if the ribbons move).

Stereo camera poses (inches relative to base_link, converted to meters below):
  left  (CAD cam 2): X=1.467, Y=+5.185, Z=4.0, yaw toe-in -10 deg
  right (CAD cam 1): X=1.467, Y=-5.185, Z=4.0, yaw toe-in +10 deg

    ros2 launch robot_bringup robot.launch.py
    ros2 launch robot_bringup robot.launch.py width:=3280 height:=2464 fps:=21
    ros2 launch robot_bringup robot.launch.py vision:=false   # skip vision node
    ros2 launch robot_bringup robot.launch.py model:=FastSAM-s.pt
    ros2 launch robot_bringup robot.launch.py backend:=yolo model:=yolov8n.pt
"""

import math
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# CAD inches -> meters. Y+ is robot left (ROS), so CAD cam2=left, cam1=right.
_IN2M = 0.0254
_CAM_X = 1.467 * _IN2M   # 0.03726 m
_CAM_Y = 5.185 * _IN2M   # 0.13170 m
_CAM_Z = 4.0 * _IN2M     # 0.10160 m
_TOE_IN = math.radians(10.0)

# camera_link (X fwd, Y left, Z up) -> optical (Z fwd, X right, Y down)
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
    """base_link -> *_camera_link -> *_camera_optical_frame (matches gscam frame_id)."""
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


def _csi_pipeline(sensor_id, width, height, fps):
    """gstreamer pipeline for one Jetson CSI camera, ending in BGR for gscam."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR"
    )


def _camera_node(name, sensor_id, width, height, fps):
    # Calibrations live in ~/.ros/camera_info/<camera_name>.yaml
    # (left_camera.yaml / right_camera.yaml). Explicit URL so gscam2 always
    # publishes real intrinsics on <cam>/camera_info for stereo triangulation.
    cam_name = f"{name}_camera"
    info_path = os.path.expanduser(f"~/.ros/camera_info/{cam_name}.yaml")
    return Node(
        package="gscam2",
        executable="gscam_main",
        name=f"{name}_camera",
        namespace=name,
        output="screen",
        parameters=[{
            "gscam_config": _csi_pipeline(sensor_id, width, height, fps),
            "camera_name": cam_name,
            "camera_info_url": f"file://{info_path}",
            "frame_id": f"{name}_camera_optical_frame",
            "use_gst_timestamps": True,
        }],
        # gscam2 publishes camera/image_raw; expose as /<name>/image_raw
        remappings=[
            ("camera/image_raw", "image_raw"),
            ("camera/camera_info", "camera_info"),
        ],
    )


def _spawn_cameras(context, *_args, **_kwargs):
    # Resolve substitutions to concrete strings before building the pipelines.
    width = LaunchConfiguration("width").perform(context)
    height = LaunchConfiguration("height").perform(context)
    fps = LaunchConfiguration("fps").perform(context)
    # NOTE: sensor-id mapping depends on which ribbon plugs into which CSI slot.
    # Physical left cam is sensor-id 1, physical right is sensor-id 0 on this Jetson.
    return [
        _camera_node("left", 1, width, height, fps),
        _camera_node("right", 0, width, height, fps),
    ]


def generate_launch_description():
    port = LaunchConfiguration("port")

    arm_launch = PathJoinSubstitution(
        [FindPackageShare("robot_arm"), "launch", "arm.launch.py"])
    vision_launch = PathJoinSubstitution(
        [FindPackageShare("robot_vision"), "launch", "vision.launch.py"])

    return LaunchDescription([
        DeclareLaunchArgument("width", default_value="1280"),
        DeclareLaunchArgument("height", default_value="720"),
        DeclareLaunchArgument("fps", default_value="30"),
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("vision", default_value="true",
                              description="start the vision detection node"),
        DeclareLaunchArgument(
            "backend",
            default_value="fastsam",
            description="fastsam (class-agnostic), blob, or yolo",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="FastSAM-s.pt",
            description="FastSAM-s.pt with backend:=fastsam, or yolov8n.pt with backend:=yolo",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([arm_launch]),
            launch_arguments={"port": port}.items(),
        ),

        *(_camera_tfs()),
        OpaqueFunction(function=_spawn_cameras),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([vision_launch]),
            launch_arguments={
                "backend": LaunchConfiguration("backend"),
                "model": LaunchConfiguration("model"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("vision")),
        ),
    ])
