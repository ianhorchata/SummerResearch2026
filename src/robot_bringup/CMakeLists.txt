cmake_minimum_required(VERSION 3.8)
project(robot_bringup)

find_package(ament_cmake REQUIRED)

# Install the launch directory so it is found at runtime.
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
