cmake_minimum_required(VERSION 3.10)
project(Project4)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Task 2, 3, 4: Camera Calibration
add_executable(calibrate calibrate.cpp)
target_link_libraries(calibrate ${OpenCV_LIBS})

# Task 5, 6: AR System (Pose + 3D object rendering)
add_executable(ar_system ar_system.cpp)
target_link_libraries(ar_system ${OpenCV_LIBS})

# Task 7: Robust Feature Detection
add_executable(robust_features robust_features.cpp)
target_link_libraries(robust_features ${OpenCV_LIBS})
