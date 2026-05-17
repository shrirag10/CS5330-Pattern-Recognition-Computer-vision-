cmake_minimum_required(VERSION 3.10)
project(da2-examples)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.26.0)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(da2-example da2-example.cpp)
target_link_libraries(da2-example ${OpenCV_LIBS} onnxruntime)
set_target_properties(da2-example PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)

add_executable(da2-video da2-video.cpp)
target_link_libraries(da2-video ${OpenCV_LIBS} onnxruntime)
set_target_properties(da2-video PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)

configure_file(${CMAKE_SOURCE_DIR}/model_fp16.onnx ${CMAKE_BINARY_DIR}/model_fp16.onnx COPYONLY)
