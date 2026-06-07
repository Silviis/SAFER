cmake_minimum_required(VERSION 3.10)
project(cmd_vel_flight_plugin)

find_package(ament_cmake REQUIRED)

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-math7 REQUIRED)

add_library(CmdVelFlightPlugin SHARED
  src/CmdVelFlightPlugin.cc
)

target_link_libraries(CmdVelFlightPlugin
  gz-sim8::gz-sim8
  gz-plugin2::gz-plugin2
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  gz-math7::gz-math7
)

install(
  TARGETS CmdVelFlightPlugin
  LIBRARY DESTINATION lib
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()