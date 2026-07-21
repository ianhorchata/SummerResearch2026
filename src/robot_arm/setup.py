from setuptools import find_packages, setup
import os
from glob import glob

package_name = "robot_arm"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # install launch files so `ros2 launch robot_arm ...` works
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="i-horch",
    maintainer_email="horchi@lafayette.edu",
    url="https://github.com/i-horch",
    description="ROS 2 driver node for a Hiwonder/LewanSoul LX serial-bus servo arm.",
    license="MIT",
    entry_points={
        "console_scripts": [
            # `ros2 run robot_arm arm_node`
            "arm_node = robot_arm.arm_node:main",
            # `ros2 run robot_arm sim_arm_bridge` (Gazebo stand-in for arm_node)
            "sim_arm_bridge = robot_arm.sim_arm_bridge:main",
            "robot_description_broadcaster = robot_arm.robot_description_broadcaster:main",
            "grab_sequence = robot_arm.grab_sequence:main",
        ],
    },
)
