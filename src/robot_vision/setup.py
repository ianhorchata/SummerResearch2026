from setuptools import find_packages, setup
import os
from glob import glob

package_name = "robot_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # install launch files so `ros2 launch robot_vision ...` works
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="i-horch",
    maintainer_email="horchi@lafayette.edu",
    url="https://github.com/i-horch",
    description="On-demand class-agnostic detection over gscam2 camera streams.",
    license="MIT",
    entry_points={
        "console_scripts": [
            # `ros2 run robot_vision vision_node`
            "vision_node = robot_vision.vision_node:main",
            "ground_pose_node = robot_vision.ground_pose_node:main",
            "drive_to_object = robot_vision.drive_to_object:main",
        ],
    },
)
