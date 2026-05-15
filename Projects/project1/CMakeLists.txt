cmake_minimum_required(VERSION 3.10)
project(Project1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(project1 main.cpp)
target_link_libraries(project1 ${OpenCV_LIBS})

add_executable(imgDisplay imgDisplay.cpp)
target_link_libraries(imgDisplay ${OpenCV_LIBS})

add_executable(vidDisplay vidDisplay.cpp)
target_link_libraries(vidDisplay ${OpenCV_LIBS})

add_executable(vidDisplayGray vidDisplayGray.cpp)
target_link_libraries(vidDisplayGray ${OpenCV_LIBS})
