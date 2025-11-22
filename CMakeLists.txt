cmake_minimum_required(VERSION 3.18)
project(MambaS6Plugin LANGUAGES CXX CUDA)

# 設定 C++ 標準
set(CMAKE_CXX_STANDARD 17)

# 設定 TensorRT 路徑 (Jetson 通常在 /usr/lib/aarch64-linux-gnu/)
# 如果找不到，請根據實際路徑修改
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu /usr/local/cuda/include)
find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)

# CUDA 設定 - 針對 Jetson Orin Nano (Ampere sm_87)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# 包含目錄
include_directories(
    ${CMAKE_SOURCE_DIR}/src/plugin
    ${CMAKE_SOURCE_DIR}/src/kernel
    ${TENSORRT_INCLUDE_DIR}
    /usr/local/cuda/include
)

# 來源檔案
set(SOURCES
    src/plugin/MambaS6Plugin.cpp
    src/kernel/mamba_s6_kernel.cu
)

# 建立共享庫
add_library(mamba_plugin SHARED ${SOURCES})

# 連結庫
target_link_libraries(mamba_plugin 
    ${TENSORRT_LIB}
    cudart
)

message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Lib: ${TENSORRT_LIB}")