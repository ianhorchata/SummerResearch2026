"""Launch on-demand class-agnostic detection (+ optional ground-pose node).

Assumes cameras are already publishing (e.g. via
`ros2 launch robot_bringup robot.launch.py`). This never opens the CSI cameras.

    ros2 launch robot_vision vision.launch.py
    ros2 launch robot_vision vision.launch.py backend:=blob
    ros2 launch robot_vision vision.launch.py backend:=fastsam model:=FastSAM-s.pt
    ros2 launch robot_vision vision.launch.py backend:=yolo model:=yolov8n.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    backend = LaunchConfiguration("backend")
    model = LaunchConfiguration("model")
    confidence = LaunchConfiguration("confidence")
    device = LaunchConfiguration("device")
    imgsz = LaunchConfiguration("imgsz")
    save_debug_images = LaunchConfiguration("save_debug_images")
    debug_image_dir = LaunchConfiguration("debug_image_dir")
    use_approx_intrinsics = LaunchConfiguration("use_approx_intrinsics")

    return LaunchDescription([
        DeclareLaunchArgument(
            "backend",
            default_value="fastsam",
            description="fastsam (class-agnostic), blob (classical), or yolo",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="FastSAM-s.pt",
            description="FastSAM-s.pt / FastSAM-x.pt, or a YOLO .pt/.engine",
        ),
        DeclareLaunchArgument("confidence", default_value="0.4"),
        DeclareLaunchArgument(
            "device",
            default_value="",
            description="'' => auto; set cuda:0 to force Jetson GPU",
        ),
        DeclareLaunchArgument(
            "imgsz",
            default_value="640",
            description="inference size; raise to 1024 for FastSAM quality",
        ),
        DeclareLaunchArgument(
            "save_debug_images",
            default_value="true",
            description="write raw+annotated JPEGs on each /vision/detect call",
        ),
        DeclareLaunchArgument(
            "debug_image_dir",
            default_value="~/ros2_ws/vision_debug",
        ),
        DeclareLaunchArgument(
            "use_approx_intrinsics",
            default_value="false",
            description="ground_pose_node: use approx K instead of camera_info",
        ),
        Node(
            package="robot_vision",
            executable="vision_node",
            name="vision_node",
            output="screen",
            parameters=[{
                "cameras": ["left", "right"],
                "backend": backend,
                "model": model,
                "confidence": confidence,
                "device": device,
                "imgsz": ParameterValue(imgsz, value_type=int),
                "save_debug_images": ParameterValue(
                    save_debug_images, value_type=bool),
                "debug_image_dir": debug_image_dir,
            }],
        ),
        Node(
            package="robot_vision",
            executable="ground_pose_node",
            name="ground_pose_node",
            output="screen",
            parameters=[{
                "use_approx_intrinsics": ParameterValue(
                    use_approx_intrinsics, value_type=bool),
            }],
        ),
    ])
