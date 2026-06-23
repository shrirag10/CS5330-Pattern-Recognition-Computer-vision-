cmake_minimum_required(VERSION 3.10)
project(Project3)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(recognition main.cpp threshold.cpp cleanup.cpp segment.cpp features.cpp utilities.cpp)
target_link_libraries(recognition ${OpenCV_LIBS})

add_executable(train train.cpp threshold.cpp cleanup.cpp segment.cpp features.cpp utilities.cpp)
target_link_libraries(train ${OpenCV_LIBS})

add_executable(evaluate evaluate.cpp threshold.cpp cleanup.cpp segment.cpp features.cpp utilities.cpp)
target_link_libraries(evaluate ${OpenCV_LIBS})
