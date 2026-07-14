# SAM2-TensorRT: Real-time Video Object Segmentation & Multi-Object Tracking

> **Single target ~30ms/frame (~33 FPS) on NVIDIA A10 — adapted for MOT, fully FP16, end-to-end GPU pipeline.**

High-performance C++ inference of Meta's [SAM2](https://github.com/facebookresearch/segment-anything-2) on NVIDIA GPUs, optimized to the metal with TensorRT.

---

## 🎬 Demo

**Single Target Tracking** — box prompt, real-time mask + bounding box at ~33 FPS:

<video src="assets/demo/single_track_demo.mp4" controls muted autoplay loop width="100%"></video>

**Multi Target Tracking** — multiple objects simultaneously:

<video src="assets/demo/multi_track_demo.mp4" controls muted autoplay loop width="100%"></video>

## 🚀 Performance

| GPU | Mode | Precision | Latency | FPS | Resolution |
|-----|------|-----------|---------|-----|------------|
| **NVIDIA A10** | Single target | **FP16** | **~30ms** | **~33 FPS** | 1920×1080 |
| NVIDIA A10 | Warm-up first frame | FP16 | ~45ms | ~22 FPS | 1920×1080 |

> End-to-end: OpenCV BGR input → GPU preprocess → 6× TensorRT engines → mask → bounding box. Including all pre/post-processing.

## 🔧 Performance Optimizations

- **Full FP16** across all 6 TensorRT engines — no INT8 calibration needed
- **GPU memory pool reuse** — all intermediate buffers pre-allocated, zero runtime allocation
- **Pinned (page-locked) memory** — minimized CPU↔GPU transfer latency
- **CUDA stream async preprocessing** — resize, normalize, BGR2RGB all on GPU
- **Combined Image Decoder** — merged two-stage decoder into single TRT engine pass
- **TBB-parallel tensor operations** — CPU-side memory bank assembly with `#pragma omp parallel for`
- **IoU-gated memory bank** — only high-confidence frames enter history, saving memory attention cost
- **GPU-based bounding box extraction** — `cv::cuda::threshold` + `findNonZero` entirely on GPU
- **卡尔曼滤波目标选择优化** — 8-D Kalman 预测框与 SAM 输出 mask 的 IoU 联合评分，抑制异常 mask 输出
- **自适应关键帧策略** — 根据目标运动状态动态决定 memory bank 写入频率，减少冗余存储

## 🏗 Architecture

```
Frame(BGR) → GPU Preprocess → Image Encoder → Memory Attention → Image Decoder → PostProcess → Mask+BBox
                                    ↑                ↑                  |
                                    |        ┌───────┘                  |
                                    |        |                          ↓
                              Memory Bank ← Memory Encoder ←───────────┘
                              (IoU-gated)
```

**6 TensorRT Engines:**

| # | Engine | Input | Output |
|---|--------|-------|--------|
| 1 | `image_encoder` | [1,3,1024,1024] | pix_feat, high_res_feat0/1, vision_feats, vision_pos_embed |
| 2 | `memory_attention` | vision features + memory bank | [B, 256, 64, 64] image_embed |
| 3 | `image_decoderStart` | point prompt + image_embed | multi-masks, multi-tokens (optional) |
| 4 | `image_decoderEnd` | selected mask token | final mask (optional) |
| 5 | `image_decoder` | point prompt + image_embed | **unified output** (recommended) |
| 6 | `memory_encoder` | mask + pix_feat | memory features |

## 📋 Features

**Tracking:**
- Box or point prompt initialization
- Per-target 8-D Kalman filter (x, y, a, h, vx, vy, va, vh)
- 卡尔曼滤波器选择优化 — 8-D Kalman 预测框与 SAM mask 的 IoU 联合评分，提升遮挡/形变场景鲁棒性
- 关键帧选取策略优化 — 基于 IoU 变化率的自适应关键帧插入，平衡追踪质量与内存开销
- Smart memory bank — IoU threshold > 0.4
- Multi-target support — per-object memory banks with fallback borrowing

**Engineering:**
- Thread-safe singleton
- TensorRT 8.x / 10.x API compatibility (compile-time switch)
- Fixed-size circular queue memory bank
- Debug timing macros (`TIME_DEBUG`, `ALL_TIMEDEBUG`)

## 🚀 Quick Start

### Prerequisites

| Dependency | Minimum | Tested |
|-----------|---------|--------|
| CUDA | 12.0 | 12.5 |
| TensorRT | 8.6 | 10.8.0 |
| cuDNN | 9.x | 9.2 |
| OpenCV | 4.8 (with CUDA) | 4.10 |
| ONNX Runtime | 1.18.0 (GPU) | 1.18.1 |
| TBB | 2021.x | 2021.13 |
| CMake | 3.25 | 3.30 |
| GCC | 9.x (C++14) | 14.1 |

### Environment Setup

```bash
export TENSORRT_ROOT=/path/to/TensorRT-10.x
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-x.x.x
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda           # auto-detected if omitted
```

## Get TensorRT Engine Files

TensorRT 10.x required. Download ONNX files from the Releases page of this repo. Two variants are available: single-target tracking and multi-target tracking.

Example:
1. Download the desired ONNX variant
2. Convert each ONNX file to a TensorRT engine:
   ```bash
   ./trtexec \
       --onnx=/memory_attention7_16.onnx \
       --saveEngine=memory_attention7_16.engine \
       --fp16
   ```
   fp16_small_singleObj:
        ./trtexec --onnx=image_encoder.onnx --saveEngine=image_encoder.engine --fp16
        ...

   fp16_small_motObj
        ./trtexec --onnx=image_encoder.onnx --saveEngine=image_encoder.engine --fp16 \
            --minShapes=batch_size:1 \
            --optShapes=batch_size:2 \
            --maxShapes=batch_size:10

        ./trtexec \
        --onnx=memory_attention7_16.onnx \
        --saveEngine=memory_attention7_16.engine \
        --fp16 \
        --minShapes=current_vision_feat:1x256x64x64,current_vision_pos_embed:4096x1x256,memory_0:16x1x256,memory_1:7x1x64x64x64,memory_pos_embed:28736x1x64 \
        --optShapes=current_vision_feat:4x256x64x64,current_vision_pos_embed:4096x4x256,memory_0:16x4x256,memory_1:7x4x64x64x64,memory_pos_embed:28736x4x64 \
        --maxShapes=current_vision_feat:10x256x64x64,current_vision_pos_embed:4096x10x256,memory_0:16x10x256,memory_1:7x10x64x64x64,memory_pos_embed:28736x10x64

        ./trtexec \
        --onnx=image_decoder.onnx \
        --saveEngine=image_decoder.engine \
        --fp16 \
        --minShapes=point_coords:1x2x2,point_labels:1x2,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128 \
        --optShapes=point_coords:4x2x2,point_labels:4x2,image_embed:4x256x64x64,high_res_feats_0:4x32x256x256,high_res_feats_1:4x64x128x128 \
        --maxShapes=point_coords:10x2x2,point_labels:10x2,image_embed:10x256x64x64,high_res_feats_0:10x32x256x256,high_res_feats_1:10x64x128x128

        ./trtexec --onnx=memory_encoder.onnx --saveEngine=memory_encoder.engine --fp16 --minShapes=mask_for_mem:1x1x1024x1024,pix_feat:1x256x64x64 --optShapes=mask_for_mem:4x1x1024x1024,pix_feat:4x256x64x64 --maxShapes=mask_for_mem:10x1x1024x1024,pix_feat:10x256x64x64

        注意：如果转trt有INT32或INT64的数值问题，那么将出问题的onnx，先用repo中的process.py处理下 仓库中的onnx均是已处理过的

3. Place all engine files under `models/fp16_small_singleObj/` (or your custom path)



### Build

```bash
cmake -B build
cmake --build build -j$(nproc)
# Output: bin/SAM2  +  libSAM2_lib.so
```

### Run

```bash
# Single image inference (SingleTrack)
./bin/SAM2 image test/photo.jpg 100 200 150 300

# Video single target (SingleTrack) 50 = video frame number
./bin/SAM2 video test/input.mp4 100 200 150 300 50

# Video multi target (MultiTrack) 50 = video frame number
./bin/SAM2 mot test/input.mp4 100 200 150 300 400 500 200 180 50
```

### API

```cpp
#include "SAM2.h"

auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();

// ===========================
// SingleTrack 单目标追踪
// ===========================

// step1: initialize 构建引擎
sam2.initialize(engine_paths, TrackerBySAM2::SingleTrack, /*gpuId=*/0);

// step2: setparms 设定单目标参数
std::vector<TrackerBySAM2::ParamsSam2> parms;
parms.push_back({/*type=*/0, cv::Rect{x, y, w, h}, {/*point=*/0, 0}});
sam2.setparms(parms);

// step3: 逐帧 inference
for (auto& frame : frames) {
    sam2.inference(frame);
    cv::Rect bbox = sam2.LastRect;
}

// ===========================
// MultiTrack 多目标追踪
// ===========================

sam2.initialize(engine_paths, TrackerBySAM2::MultiTrack, /*gpuId=*/0);

std::vector<TrackerBySAM2::ParamsSam2> multi_parms;
for (auto& bbox : init_bboxes) {
    multi_parms.push_back({0, bbox, {0, 0}});
}
sam2.setparms(multi_parms);

for (auto& frame : frames) {
    sam2.inference(frame);
    cv::Rect bbox = sam2.LastRect;
}
```

## 📦 Models

TensorRT engine files are required but **not included** in this repo due to size.

- **Pre-built engines**: Contact the author for download
- **Export your own**: See `2onnx_tools/SAM2Export-main/` for ONNX → TensorRT scripts

## 📁 Structure

```
sam2_cpp/
├── src/
│   ├── main.cpp           # Demo entry point
│   └── SAM2.cpp           # Core engine (~3400 lines)
├── include/
│   ├── SAM2.h             # TensorRTInference + Sam2Singleton
│   ├── Model.h            # Base class + FixedSizeQueue
│   └── bytetrack/         # Kalman filter, STrack, LAPJV
├── 2onnx_tools/           # ONNX/TensorRT export tools
├── CMakeLists.txt
├── config.cmake
├── CONTRIBUTING.md
└── LICENSE
```

## 🙏 Acknowledgments

- [SAM2](https://github.com/facebookresearch/segment-anything-2) — Meta
- [SAM2Export](https://github.com/Aimol-l/SAM2Export)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

## 📄 License

MIT. SAM2 model weights subject to Meta's license.
