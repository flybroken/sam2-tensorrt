cmake_minimum_required(VERSION 3.15)

include(config.cmake)

project(
    SAM2
    LANGUAGES CXX C
    DESCRIPTION "SAM2-TensorRT: Real-time Video Object Tracking & Segmentation"
)

# ---- 编译选项 ----
if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    message(STATUS "Build type: RelWithDebInfo")
else()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    message(STATUS "Build type: Release (-O3)")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 输出目录 ----
get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)

# ---- 头文件 ----
include_directories(${PROJECT_DIR}/include/)
include_directories(${PROJECT_DIR}/include/bytetrack/include/)

# ---- 编译宏 ----
if(TENSORRT_8_X_)
    add_definitions(-DTENSORRT_8_X)
    message(STATUS "TensorRT API: 8.x")
else()
    message(STATUS "TensorRT API: 10.x")
endif()

if(DRAW_RESULT)
    add_definitions(-DDRAW_PERSON)
    message(STATUS "Draw bounding boxes: ON")
endif()

# ---- CUDA ----
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_TOOLKIT_ROOT_DIR}")
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
endif()
find_package(CUDA REQUIRED)
message(STATUS "CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")

# ---- TensorRT ----
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
if(NOT TENSORRT_ROOT)
    message(FATAL_ERROR "TENSORRT_ROOT environment variable not set. "
                        "Example: export TENSORRT_ROOT=/path/to/TensorRT-10.x")
endif()
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)
message(STATUS "TensorRT: ${TENSORRT_ROOT}")

# ---- ONNX Runtime ----
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT environment variable not set. "
                        "Example: export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-x.x.x")
endif()
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}")
    message(FATAL_ERROR "ONNX Runtime include dir not found: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_ROOT}/lib)
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")

# ---- OpenCV ----
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs videoio highgui dnn
    OPTIONAL_COMPONENTS cudaarithm cudawarping cudaimgproc
)
message(STATUS "OpenCV: ${OpenCV_VERSION} (modules: ${OpenCV_LIBS})")

# ---- TBB ----
find_package(TBB REQUIRED)
message(STATUS "TBB: ${TBB_DIR}")

# ---- 源文件 ----
file(GLOB SRC_FILES
    CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/bytetrack/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/bytetrack/src/*.cpp"
)

# ---- 构建目标 ----
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})
add_library(${CMAKE_PROJECT_NAME}_lib SHARED ${SRC_FILES})

# ---- 链接 ----
# 主可执行文件
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OpenCV_LIBS}
    TBB::tbb
    -L${ONNXRUNTIME_ROOT}/lib -lonnxruntime
    -L${TENSORRT_ROOT}/lib -lnvinfer
    -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcudart
    -lnvonnxparser
    -lpthread
)

# 共享库 (去掉可执行文件特有的依赖)
target_link_libraries(${CMAKE_PROJECT_NAME}_lib
    ${OpenCV_LIBS}
    TBB::tbb
    -L${ONNXRUNTIME_ROOT}/lib -lonnxruntime
    -L${TENSORRT_ROOT}/lib -lnvinfer
    -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcudart
    -lpthread
)
