# 编译配置选项
# 使用方式: cmake -B build -DDEBUG_MODE=ON ...

# 调试模式 (RelWithDebInfo)，默认 Release + -O3
set(DEBUG_MODE    OFF CACHE BOOL "Build with debug symbols (RelWithDebInfo)")

# TensorRT 8.x API 兼容模式，默认使用 10.x API
set(TENSORRT_8_X_ OFF CACHE BOOL "Use TensorRT 8.x API (default: 10.x)")

# 在输出图像上绘制目标框
set(DRAW_RESULT   OFF CACHE BOOL "Draw person bounding boxes on output images")
