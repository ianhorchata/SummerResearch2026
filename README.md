# Create 3 + Jetson Orin Nano robot (ROS 2 Jazzy)

ROS 2 workspace integrating an iRobot Create 3, a Hiwonder LX serial-bus servo
arm, and two CSI cameras on an NVIDIA Jetson Orin Nano.

## What's here

```
ros2_ws/
├── src/
│   ├── robot_arm/          Hiwonder LX arm driver + grab sequence
│   ├── robot_interfaces/   custom services (MoveJoints, DetectObjects, …)
│   ├── robot_vision/       YOLO detect + stereo ground poses + drive-to-object
│   └── robot_bringup/      launch: arm + CSI cams + optional vision
├── sim/
│   ├── create3_description/              hardware-oriented URDF + meshes
│   └── create3_urdf_assem_description/   Gazebo/RViz arm assembly URDF
├── camera_info/            left/right CSI intrinsics
└── config/                 Fast DDS helper XML
```

Clone camera deps into `src/` (not checked in): `gscam2` and `ros2_shared` — see setup below.
`sim/create3_sim` (iRobot Gazebo stack) is intentionally not in this repo yet.

The **Create 3 is not in this workspace** on purpose — it runs ROS 2 itself and
publishes its topics over the Jetson<->Create network. You just consume them
(`/odom`, `/battery_state`, `/imu`, `/cmd_vel`, ...).

## The ROS 2 graph

```
create3        -> /odom /battery_state /imu ... (and subscribes /cmd_vel)
left_camera    -> /left/image_raw   /left/camera_info
right_camera   -> /right/image_raw  /right/camera_info
arm_node       -> /arm/joint_states (sensor_msgs/JointState, RADIANS)
               -> /arm/health       (robot_interfaces/ServoHealth)
               <- /arm/joint_command (sensor_msgs/JointState, RADIANS)
               services: /arm/scan /arm/home /arm/set_torque /arm/move_joints
vision_node    <- /left/image_raw /right/image_raw (subscribes only)
               service: /vision/detect (robot_interfaces/DetectObjects)
```

`arm_node` **scans the serial bus at startup** and auto-discovers every servo id
that responds (using `HiwonderServoBus.scan`). Re-scan any time with
`ros2 service call /arm/scan std_srvs/srv/Trigger`.

## One-time setup (on the Jetson)

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install python3-serial          # for pyserial
sudo usermod -aG dialout $USER           # serial access without sudo; re-login after

# CSI cameras are published with gscam2 (GStreamer -> ROS). If not packaged:
cd ~/ros2_ws/src
git clone https://github.com/clydemcqueen/gscam2.git
git clone https://github.com/ptrmu/ros2_shared.git   # gscam2 build dependency
sudo apt install libgstreamer1.0-dev gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad ros-jazzy-camera-calibration-parsers

# Object detection (robot_vision) uses vision_msgs + cv_bridge + Ultralytics YOLO:
sudo apt install ros-jazzy-vision-msgs ros-jazzy-cv-bridge
pip3 install --user --break-system-packages ultralytics
# Ultralytics may pull packages that break ROS / colcon. Pin these:
pip3 install --user --break-system-packages 'setuptools==70.3.0' 'numpy<2'
# (setuptools>=72 breaks `colcon --symlink-install`; numpy>=2 breaks
#  apt-built cv_bridge / matplotlib which were compiled against NumPy 1.x)
```

## Build

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash      # do this in every new shell (or add to ~/.bashrc)
```

Build `robot_interfaces` first if you hit an ordering issue:
`colcon build --packages-select robot_interfaces` then build the rest.

## Run

Just the arm:

```bash
ros2 launch robot_arm arm.launch.py            # uses /dev/ttyACM0
ros2 launch robot_arm arm.launch.py port:=/dev/ttyACM1
```

Everything (arm + both CSI cameras):

```bash
ros2 launch robot_bringup robot.launch.py
ros2 launch robot_bringup robot.launch.py width:=1920 height:=1080 fps:=30
```

## Try it

```bash
# confirm Jetson <-> Create 3 ROS comms work (no code needed):
ros2 topic list
ros2 topic echo /battery_state

# watch the arm report its joint angles:
ros2 topic echo /arm/joint_states
ros2 topic echo /arm/health

# move servos 1 and 2 to 120 deg over 1 s:
ros2 service call /arm/move_joints robot_interfaces/srv/MoveJoints \
  "{ids: [1, 2], positions_deg: [120.0, 120.0], time_ms: 1000}"

# center everything / release torque:
ros2 service call /arm/home std_srvs/srv/Trigger
ros2 service call /arm/set_torque std_srvs/srv/SetBool "{data: false}"

# view a camera (on a machine with a display):
ros2 run rqt_image_view rqt_image_view   # pick /left/image_raw

# run object detection on the latest frame from a camera:
ros2 service call /vision/detect robot_interfaces/srv/DetectObjects "{camera: 'left'}"
ros2 service call /vision/detect robot_interfaces/srv/DetectObjects \
  "{camera: 'right', confidence: 0.4}"
```

## Object detection (`robot_vision`)

`vision_node` runs detection **on request only** (a service), because inference
is expensive and you rarely want it on every frame. It does **not** open the
cameras — gscam2 already owns them, and the Jetson Argus source allows one
consumer per sensor. Instead it subscribes to `/left/image_raw` and
`/right/image_raw`, caches the newest frame from each, and runs a YOLO model on
that frame when `/vision/detect` is called. The response is a
`vision_msgs/Detection2DArray` (labels + confidences + pixel bounding boxes).

By default each successful detect also writes two JPEGs under
`~/ros2_ws/vision_debug/`:
`detect_<cam>_<timestamp>_raw.jpg` and `..._annotated.jpg` (boxes drawn).
Disable with `save_debug_images:=false`.

```bash
# cameras must be publishing first (robot.launch.py does this):
ros2 launch robot_vision vision.launch.py
ros2 launch robot_vision vision.launch.py model:=/home/ianho/models/best.engine
ros2 launch robot_vision vision.launch.py save_debug_images:=false
```

- **Model:** defaults to `yolov8n.pt` (downloads on first use). Point `model:=`
  at your own `.pt`, or — for real speed on the Jetson — export a TensorRT
  engine (`yolo export model=best.pt format=engine`) and pass the `.engine`.
- **GPU:** set `device:=cuda:0` to force the Jetson GPU.
- **Latency:** the service call blocks until inference finishes. If you later
  need progress/cancel or run heavier models, switch it to a ROS 2 action.

## Notes / gotchas

- **Units:** `joint_states` / `joint_command` use **radians** (ROS convention,
  so this plugs into `robot_state_publisher`, RViz, MoveIt later). The
  `move_joints` service uses **degrees** for convenience. The underlying servos
  are 0–1000 units = 0–240°.
- **Single serial bus:** the node runs single-threaded on purpose so position
  reads and commands never collide on the half-duplex bus.
- **Camera sensor-id:** `robot.launch.py` maps CSI slot 0 -> `/left`, slot 1 ->
  `/right`. Swap the `0`/`1` in `_spawn_cameras` if they come out reversed.
- **DDS across machines:** make sure the Jetson and any host use the same
  `ROS_DOMAIN_ID` and are on the same subnet to see each other's topics.

## Visualizing the arm in RViz

`create3_description` holds the URDF (Fusion 360 export) and a display launch.
Three nodes cooperate: `robot_state_publisher` (URDF -> TF), a joint-state
source, and `rviz2`.

**Step 1 — see the model + move it by hand (no hardware needed):**

```bash
colcon build --symlink-install && source install/setup.bash
ros2 launch create3_description display.launch.py        # use_gui:=true (default)
```

RViz opens with the robot model; a slider window lets you drive each movable
joint. This confirms the URDF, meshes, and TF tree are correct.

**Step 2 — drive the model from the live arm:**

```bash
# terminal A: the arm (publishes /arm/joint_states)
ros2 launch robot_arm arm.launch.py
# terminal B: RViz fed by the live arm
ros2 launch create3_description display.launch.py use_gui:=false
```

In `use_gui:=false` mode a plain `joint_state_publisher` merges
`/arm/joint_states` and defaults every other joint (wheels, etc.) to 0.

### Matching servo ids to URDF joint names (required for Step 2)

The URDF joints have Fusion-generated names with spaces (e.g. `Revolute 14`),
but `arm_node` publishes `servo_<id>` by default — they won't line up until you
map them with the `joint_name_map` parameter:

```
"<id>:<joint name>[:<center_units>[:<sign>]]"
  center_units : servo position (0-1000) that equals the URDF joint's zero pose
                 (defaults to 500, the servo midpoint)
  sign         : +1 or -1, flip if the model rotates the wrong way
```

Example — servo id 1 is `Revolute 14`, centered, reversed:

```bash
ros2 run robot_arm arm_node --ros-args \
  -p "joint_name_map:=['1:Revolute 14:500:-1']"
```

Tip to calibrate: launch Step 2, watch the model, then tweak `sign` (direction)
and `center_units` (offset) until the on-screen arm matches the real one. Find
which servo id is which joint with `ros2 service call /arm/scan
std_srvs/srv/Trigger` and by jogging one servo at a time.

> Note: the URDF currently models only one revolute joint (`Revolute 14`) plus
> fixed supports. As you add the remaining arm joints in Fusion/xacro, add a
> matching `joint_name_map` entry for each.

## Suggested next steps

1. Add `robot_state_publisher` + a URDF so RViz shows the arm + base together.
2. Add a teleop node that maps a joystick to `/cmd_vel` (drive) and
   `/arm/joint_command` (arm).
3. Calibrate the cameras and feed `camera_info` for stereo depth.
