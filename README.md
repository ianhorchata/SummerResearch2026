# Lafayette College Summer Research 2026
## Create3 + Jetson Orin Nano Object Detection and Pickup

This research project was made possible and supervised by Professor Yu.
The goal of this project was to develop a mobile base and vision system capable of recognizing and picking up small objects. The project utilizes iRobot's [Create3](https://iroboteducation.github.io/create3_docs/) for the mobile base, a [Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) as the compute board, a custom-made 3D-printed arm, and a 3d printed basket for items to be dropped into.

<img width="5712" height="4284" alt="IMG_0010" src="https://github.com/user-attachments/assets/5d74d311-ccb2-4ba8-81e6-0607e5221b99" />

## Videos

[![Create3 Succesful Grab](https://img.youtube.com/vi/UcSSc0AryHc/0.jpg)](https://www.youtube.com/watch?v=UcSSc0AryHc)

[![Create3 Succesful Sweep](https://img.youtube.com/vi/dUKRa-kdFyE/0.jpg)](https://www.youtube.com/watch?v=dUKRa-kdFyE)

## Hardware

The project has a [Create3](https://iroboteducation.github.io/create3_docs/). The Create3 has a 26Wh battery capable of delivering a nominal 14.4 volts at a maximum of 2 amps, for ~28 watts of power. This is enough to power the Jetson compute board, which draws ~25W of power. The arm was initially designed with normal PWM servos in mind. However, these servos are open-loop, require three wires each, and would not be accurate enough for this project. Instead, the arm uses four [LX-224 Serial Bus Servos](https://www.amazon.com/LewanSoul-Connectors-Equipped-Position-Temperature/dp/B0817X3Z3W). These servos use a UART half-duplex serial bus to communicate with the Jetson via the BusLinkerV3 board. The servos broadcast their position with 0.24 degree resolution, allowing for accurate closed-loop control. The BusLinker board can support up to 253 servos on the same bus, more than enough to control the four servos used on the arm. The servos run at 6-8.4volts and draw upwards of 3 amps at stall, for a theoretical maximum of 33.6 watts of power. While the servos will rarely be operated at stall simultaneously, serious power draw will occur when all four servos are moving simultaneously. This is too much power for the Create3 to provide on top of the Jetson demands. Therefore, the arm requires a dedicated battery for power. The battery is a two-cell 52000mAh 50c LiPo. As a two-cell, it has a nominal voltage of 7.4 volts and at 50c can output more than 200amps of power, way more than the servos need. (Pro tip: when replacing the battery connector, do not cut both terminals of the battery at once. That will cause a short through the wire stripper and will blow it up. Ask me how I know)

The vision system uses two IMX Arducam 219 8MP cameras for stereoscopic vision. The camera placement was chosen to allow for a nearly unobstructed view of the ideal grabbing area. In retrospect, giving the cameras a higher vantage point would have been beneficial. 

## Design Considerations

The camera's FOV was modelled as a rectangular pyramid to show what the camera would see. In SolidWorks, you can add a camera to a model and literally see through the camera's viewpoint; however, as far as I know, Fusion lacks that capability. The camera positions were chosen to fit within Create3's frame without obstruction. The camera's FOV overlap about 10 inches in front of the Create3, which defines the minimum distance the arm should be when it grabs an object, as the object must be visible in both cameras for stereo vision to function.

The arm uses four LX-224 servos. These servos have a rated torque of 20 kg $$\cdot$$ cm. In my experience, servo torque is almost always less than advertised. A better number is likely 10-15 kg $$\cdot$$ cm. The highest moment occurs on the shoulder joint when the arm is fully extended horizontally. To grab objects in range of the cameras, each section of arm needed to be ~5 inches. This corresponds to a moment of 10.8 kg $$\cdot$$ cm. Initial testing using a DC power supply proved worrisome, but when powered by the battery, the arm had more than enough torque. 

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

The Create3 runs its own ROS2 nodes. For more information on what topics and services the Create3 provides, please see the Create3 [API documentation](https://iroboteducation.github.io/create3_docs/api/ros2/). The Create3 and the Jetson communicate over a virtual Ethernet USB-C link. The Jetson and the Create3 must both use fastDDS and have the same ROS_DOMAIN_ID for UDP to succeed.

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

`vision_node` runs detection as a service, because inference
is expensive and you rarely want it on every frame. The vision node subscribes to `/left/image_raw` and
`/right/image_raw`, caches the newest frame from each, and runs a YOLO model on
that frame when `/vision/detect` is called. The default model is FastSAM to segment the image into objects. The response is a
`vision_msgs/Detection2DArray`; each bounding box has a label and a confidence score. The FastSAM algorithm has a lot of noise, and extensive filtering is required to determine what the real grabbable objects are in the frame. 

<table>
  <tr>
    <th><b>Left Camera</b></th>
    <th><b>Right Camera</b></th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9772c9ce-bc85-4fa0-9db5-2cc5ef401246" width="100%" alt="Left Camera"></td>
    <td><img src="https://github.com/user-attachments/assets/7e4f8eed-344c-4401-b417-1b6daf9d11e2" width="100%" alt="Right Camera"></td>
  </tr>
</table>

The main challenge for stereoscopic vision is finding the same object in both camera images. Once done, stereoscopic vision allows for depth perception and can accurately determine the size and location of an object. Objects were initially matched based on ground-pose estimation. However, this proved unreliable. Instead, objects were matched based on epipolar geometry. A real object match must satisfy the epipolar constraint ```p_R^T·F·p_L = 0```, where F is built from the calibrated camera intrinsics. For each object in the left image, take the bottom center of the bounding box and compute the epipolar line: ```Line = p_L@F''' and then compute the perpendicular distance from the right pixel to that line. If the two boxes depict the same 3d point, the distance should be near zero, but since there is inherent variance in how bounding boxes are drawn around the objects, an error of less than 40 pixels is tolerated. Sometimes epipolar geometry can agree even if it's not the same 3d objects. To filter against this, we assign a penalty for mismatched size/aspect ratio of the bounding box and mismatched appearance. However, the appearance penalty is often faulty when there is glare in the image.

The naive approach of matching the closest right object per left object fails when there are multiple close right objects, and can lead to incorrect assignment. To solve this, we build a cost matrix of all possible left and right matches, and run a linear sum algorithm to determine the best overall matches. This is reliably able to filter out noise and find which real-world objects are matched.

After successful matching, a 3D pose can be obtained by triangulation. An apparent size can be estimated based on the size of the bounding box and the distance of the object. Objects larger than a set threshold, further away than allowed, or not on the ground, can then be filtered out, leaving only valid grab candidates.

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
- **Latency:** the service call blocks until inference finishes.

## Notes

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

**Step 1 — see the model + move it with sliders (no hardware needed):**

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
