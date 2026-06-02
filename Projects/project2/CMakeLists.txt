cmake_minimum_required(VERSION 3.10)
project(Project2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Task 1+: content-based image retrieval
add_executable(match main.cpp)
target_link_libraries(match ${OpenCV_LIBS})
