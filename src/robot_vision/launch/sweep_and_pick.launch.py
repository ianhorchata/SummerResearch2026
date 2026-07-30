"""Launch the lawn-mower sweep + pick mission.

Assumes Create3 and ``robot_bringup`` (arm, cameras, vision) are already
running. Starts only the mission node.

    ros2 launch robot_vision sweep_and_pick.launch.py
    ros2 launch robot_vision sweep_and_pick.launch.py skip_undock:=true
    ros2 launch robot_vision sweep_and_pick.launch.py battery_low_pct:=0.15
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    area_length_m = LaunchConfiguration("area_length_m")
    area_width_m = LaunchConfiguration("area_width_m")
    lane_spacing_m = LaunchConfiguration("lane_spacing_m")
    margin_m = LaunchConfiguration("margin_m")
    scan_interval_m = LaunchConfiguration("scan_interval_m")
    clear_dock_distance_m = LaunchConfiguration("clear_dock_distance_m")
    pre_dock_approach_m = LaunchConfiguration("pre_dock_approach_m")
    dock_keepout_m = LaunchConfiguration("dock_keepout_m")
    area_yaw_offset_rad = LaunchConfiguration("area_yaw_offset_rad")
    battery_low_pct = LaunchConfiguration("battery_low_pct")
    servo_low_voltage_v = LaunchConfiguration("servo_low_voltage_v")
    servo_low_hold_sec = LaunchConfiguration("servo_low_hold_sec")
    max_picks = LaunchConfiguration("max_picks")
    skip_undock = LaunchConfiguration("skip_undock")
    skip_dock = LaunchConfiguration("skip_dock")
    control_mode = LaunchConfiguration("control_mode")
    auto_grab = LaunchConfiguration("auto_grab")
    dry_run = LaunchConfiguration("dry_run")
    settle_sec = LaunchConfiguration("settle_sec")
    still_hold_sec = LaunchConfiguration("still_hold_sec")
    sense_frames = LaunchConfiguration("sense_frames")
    sense_min_hits = LaunchConfiguration("sense_min_hits")
    lock_gate_m = LaunchConfiguration("lock_gate_m")
    heading_tol_rad = LaunchConfiguration("heading_tol_rad")
    grab_y_tol_m = LaunchConfiguration("grab_y_tol_m")
    grab_retry_max = LaunchConfiguration("grab_retry_max")

    return LaunchDescription([
        DeclareLaunchArgument(
            "area_length_m",
            default_value="2.7432",
            description=(
                "Carpet long axis (9 ft) forward from dock along post-undock heading"
            ),
        ),
        DeclareLaunchArgument(
            "area_width_m",
            default_value="1.524",
            description=(
                "Carpet short axis (5 ft) to the RIGHT of the dock; "
                "lanes step across this axis"
            ),
        ),
        DeclareLaunchArgument(
            "lane_spacing_m",
            default_value="0.30",
            description="Spacing between parallel lanes (across the 5 ft width)",
        ),
        DeclareLaunchArgument(
            "margin_m",
            default_value="0.35",
            description="Keep robot center this far inside carpet edges",
        ),
        DeclareLaunchArgument("scan_interval_m", default_value="0.40"),
        DeclareLaunchArgument("clear_dock_distance_m", default_value="0.30"),
        DeclareLaunchArgument(
            "pre_dock_approach_m",
            default_value="0.70",
            description=(
                "Drive this far into the carpet from the dock corner "
                "(and face the charger) before issuing Dock"
            ),
        ),
        DeclareLaunchArgument(
            "dock_keepout_m",
            default_value="0.90",
            description=(
                "Ignore detections within this radius (m) of area origin "
                "(dock corner) so the charger is never a pick target"
            ),
        ),
        DeclareLaunchArgument(
            "area_yaw_offset_rad",
            default_value="0.0",
            description="Add to post-undock yaw if dock faces the short side",
        ),
        DeclareLaunchArgument(
            "battery_low_pct",
            default_value="0.20",
            description="Auto-dock when Create3 battery percentage <= this",
        ),
        DeclareLaunchArgument(
            "servo_low_voltage_v",
            default_value="7.8",
            description="Servo undervoltage threshold (V)",
        ),
        DeclareLaunchArgument(
            "servo_low_hold_sec",
            default_value="10.0",
            description=(
                "Require continuous low servo voltage for this many seconds "
                "before auto-dock (ignores brief grab brownouts)"
            ),
        ),
        DeclareLaunchArgument("max_picks", default_value="20"),
        DeclareLaunchArgument(
            "skip_undock",
            default_value="false",
            description="Start sweep from current pose (already off dock)",
        ),
        DeclareLaunchArgument(
            "skip_dock",
            default_value="false",
            description="Do not call Dock at end / on abort",
        ),
        DeclareLaunchArgument(
            "control_mode",
            default_value="navigate",
            description="Pick approach mode: navigate or servo",
        ),
        DeclareLaunchArgument("auto_grab", default_value="true"),
        DeclareLaunchArgument("dry_run", default_value="false"),
        DeclareLaunchArgument(
            "settle_sec",
            default_value="0.5",
            description=(
                "Extra pause after robot is still, before camera capture (seconds)"
            ),
        ),
        DeclareLaunchArgument(
            "still_hold_sec",
            default_value="0.6",
            description=(
                "Continuous near-zero twist required before settle_sec (seconds)"
            ),
        ),
        DeclareLaunchArgument(
            "sense_frames",
            default_value="3",
            description=(
                "Stereo detects per stop (multi-frame consensus to reduce flicker)"
            ),
        ),
        DeclareLaunchArgument(
            "sense_min_hits",
            default_value="2",
            description="Object must appear in this many sense frames to count",
        ),
        DeclareLaunchArgument(
            "lock_gate_m",
            default_value="0.40",
            description=(
                "Stick to first chosen object within this odom distance (meters)"
            ),
        ),
        DeclareLaunchArgument(
            "heading_tol_rad",
            default_value="0.03",
            description=(
                "Max |atan2(y,x)| before grab (~0.03 rad ≈ 1.1 cm at 0.38 m)"
            ),
        ),
        DeclareLaunchArgument(
            "grab_y_tol_m",
            default_value="0.012",
            description=(
                "Max |y| in base_link before grab (planar IK ignores lateral)"
            ),
        ),
        DeclareLaunchArgument(
            "grab_retry_max",
            default_value="2",
            description=(
                "Extra grab attempts if post-grab verify still sees the target"
            ),
        ),
        Node(
            package="robot_vision",
            executable="sweep_and_pick",
            name="sweep_and_pick",
            output="screen",
            parameters=[{
                "area_length_m": ParameterValue(
                    area_length_m, value_type=float),
                "area_width_m": ParameterValue(
                    area_width_m, value_type=float),
                "lane_spacing_m": ParameterValue(
                    lane_spacing_m, value_type=float),
                "margin_m": ParameterValue(margin_m, value_type=float),
                "scan_interval_m": ParameterValue(
                    scan_interval_m, value_type=float),
                "clear_dock_distance_m": ParameterValue(
                    clear_dock_distance_m, value_type=float),
                "pre_dock_approach_m": ParameterValue(
                    pre_dock_approach_m, value_type=float),
                "dock_keepout_m": ParameterValue(
                    dock_keepout_m, value_type=float),
                "area_yaw_offset_rad": ParameterValue(
                    area_yaw_offset_rad, value_type=float),
                "battery_low_pct": ParameterValue(
                    battery_low_pct, value_type=float),
                "servo_low_voltage_v": ParameterValue(
                    servo_low_voltage_v, value_type=float),
                "servo_low_hold_sec": ParameterValue(
                    servo_low_hold_sec, value_type=float),
                "max_picks": ParameterValue(max_picks, value_type=int),
                "skip_undock": ParameterValue(skip_undock, value_type=bool),
                "skip_dock": ParameterValue(skip_dock, value_type=bool),
                "control_mode": control_mode,
                "auto_grab": ParameterValue(auto_grab, value_type=bool),
                "dry_run": ParameterValue(dry_run, value_type=bool),
                "settle_sec": ParameterValue(settle_sec, value_type=float),
                "still_hold_sec": ParameterValue(
                    still_hold_sec, value_type=float),
                "sense_frames": ParameterValue(sense_frames, value_type=int),
                "sense_min_hits": ParameterValue(
                    sense_min_hits, value_type=int),
                "lock_gate_m": ParameterValue(lock_gate_m, value_type=float),
                "heading_tol_rad": ParameterValue(
                    heading_tol_rad, value_type=float),
                "grab_y_tol_m": ParameterValue(grab_y_tol_m, value_type=float),
                "grab_retry_max": ParameterValue(
                    grab_retry_max, value_type=int),
            }],
        ),
    ])
