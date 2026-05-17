cmake_minimum_required(VERSION 3.10)
project(Project1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.26.0)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

# Task 1: image display
add_executable(imgDisplay imgDisplay.cpp)
target_link_libraries(imgDisplay ${OpenCV_LIBS})

# Tasks 2-11: unified video display + all filters + depth estimation
add_executable(vidDisplay vidDisplay.cpp filter.cpp faceDetect/faceDetect.cpp)
target_include_directories(vidDisplay PRIVATE faceDetect)
target_link_libraries(vidDisplay ${OpenCV_LIBS} onnxruntime)
set_target_properties(vidDisplay PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)

# Copy Haar cascade XML to build directory so the runtime path resolves
configure_file(faceDetect/haarcascade_frontalface_alt2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/haarcascade_frontalface_alt2.xml
               COPYONLY)

# Copy DA2 model to build directory
configure_file(${CMAKE_SOURCE_DIR}/da2-code/model_fp16.onnx
               ${CMAKE_CURRENT_BINARY_DIR}/model_fp16.onnx
               COPYONLY)

# OpenCV version sanity check
add_executable(project1 main.cpp)
target_link_libraries(project1 ${OpenCV_LIBS})

