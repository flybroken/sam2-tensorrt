# SAM2-TensorRT: 30ms Real-time Video Object Segmentation

> **Single target: 30ms per frame on NVIDIA A10 — fully FP16, end-to-end GPU pipeline.**

High-performance C++ inference of Meta's [SAM2](https://github.com/facebookresearch/segment-anything-2) on NVIDIA GPUs, optimized to the metal with TensorRT.

---

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
- IoU-weighted mask selection (SAM score + Kalman prediction IoU)
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

### Build

```bash
cmake -B build
cmake --build build -j$(nproc)
# Output: bin/SAM2  +  libSAM2_lib.so
```

### Run

```bash
# Single image with initial bounding box (x, y, w, h)
./bin/SAM2 image test/photo.jpg 100 200 150 300

# Video tracking + benchmark
./bin/SAM2 video test/input.mp4 100 200 150 300 50
```

### API

```cpp
#include "SAM2.h"

auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();

sam2.initialize(engine_paths, TrackerBySAM2::SingleTrack, /*gpuId=*/0);

std::vector<cv::Mat> frames;
cv::Rect init_bbox{x, y, w, h};
std::vector<cv::Rect> out;
sam2.sam2Process(frames, init_bbox, out);

// Frame-by-frame alternative:
for (auto& f : frames) {
    sam2.inference(f);
    cv::Rect r = sam2.LastRect;
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
