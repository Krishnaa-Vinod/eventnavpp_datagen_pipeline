cmake_minimum_required(VERSION 3.8)
project(slam_rgbd_event)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(event_camera_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(event_camera_codecs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(include ${OpenCV_INCLUDE_DIRS})


add_executable(visualization_node
   src/visualization_node.cpp
)

# add_executable(slam_main_node
#    src/main.cpp
# )
target_link_libraries(visualization_node Eigen3::Eigen ${OpenCV_LIBS})
ament_target_dependencies(visualization_node rclcpp std_msgs sensor_msgs event_camera_msgs cv_bridge image_transport Eigen3 event_camera_codecs)

# target_link_libraries(slam_main_node Eigen3::Eigen)
# ament_target_dependencies(slam_main_node rclcpp std_msgs sensor_msgs Eigen3)

install(TARGETS visualization_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
