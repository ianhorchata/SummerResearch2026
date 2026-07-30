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
    undistort = LaunchConfiguration("undistort")
    project_bbox_point = LaunchConfiguration("project_bbox_point")
    match_mode = LaunchConfiguration("match_mode")
    epipolar_tol_px = LaunchConfiguration("epipolar_tol_px")
    use_appearance = LaunchConfiguration("use_appearance")
    min_appearance = LaunchConfiguration("min_appearance")
    require_on_ground = LaunchConfiguration("require_on_ground")
    ground_contact_z_max = LaunchConfiguration("ground_contact_z_max")

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
            description=(
                "buffer annotated JPEGs; only write when a detect is used "
                "(or DetectObjects.save_debug:=true)"
            ),
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
        DeclareLaunchArgument(
            "undistort",
            default_value="true",
            description="ground_pose_node: undistort pixels with D before rays",
        ),
        DeclareLaunchArgument(
            "project_bbox_point",
            default_value="bottom",
            description="ground_pose_node: project bbox 'bottom' or 'center'",
        ),
        DeclareLaunchArgument(
            "match_mode",
            default_value="epipolar",
            description="stereo association: 'epipolar' or 'ground'",
        ),
        DeclareLaunchArgument(
            "epipolar_tol_px",
            default_value="40.0",
            description=(
                "max distance (px) from right point to left epipolar line "
                "(stereo-cal F is tighter; TF fallback needs ~30-40)"
            ),
        ),
        DeclareLaunchArgument(
            "use_appearance",
            default_value="true",
            description=(
                "use grayscale NCC on bbox crops in Hungarian assign cost; "
                "not included in reported match_cost"
            ),
        ),
        DeclareLaunchArgument(
            "min_appearance",
            default_value="-1.0",
            description=(
                "hard-reject matches with NCC below this; "
                "<=0 disables (appearance is soft assign cost only)"
            ),
        ),
        DeclareLaunchArgument(
            "require_on_ground",
            default_value="true",
            description=(
                "drop stereo poses whose bbox-bottom contact_z is off the floor"
            ),
        ),
        DeclareLaunchArgument(
            "ground_contact_z_max",
            default_value="0.15",
            description="max contact z (m) above ground_z to count as on-ground",
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
                "undistort": ParameterValue(undistort, value_type=bool),
                "project_bbox_point": project_bbox_point,
                "match_mode": match_mode,
                "epipolar_tol_px": ParameterValue(
                    epipolar_tol_px, value_type=float),
                "use_appearance": ParameterValue(
                    use_appearance, value_type=bool),
                "min_appearance": ParameterValue(
                    min_appearance, value_type=float),
                "require_on_ground": ParameterValue(
                    require_on_ground, value_type=bool),
                "ground_contact_z_max": ParameterValue(
                    ground_contact_z_max, value_type=float),
            }],
        ),
    ])
