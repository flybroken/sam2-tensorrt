# SAM2 C++ 输入方式（API）说明

本文说明本工程支持的几种提示（prompt）输入方式，以及各自该怎么调用。

---

## 1. 先建立一个核心认知

**模型没有独立的「框输入」。** TensorRT 引擎 `image_decoder` 只有两个提示输入：

| 输入名 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `point_coords` | `[B, N, 2]` | float32 | N 个点的坐标，值域缩放到 0–1024 |
| `point_labels` | `[B, N]` | int32 | 每个点的标签 |

- **框** = 用「两个点 + 标签 2/3」表示（左上角、右下角）
- **点** = 用「一个点 + 标签 1」表示（1=前景，0=背景）

其中 `B` = 目标数量（batch），`N` = 每个目标的点数。

---

## 2. 三种输入方式

| 输入方式 | `ParamsSam2.type` | N | `point_coords` | `point_labels` |
|---|---|---|---|---|
| **只有框** | `PromptBox` (0) | 2 | `[x0,y0, x1,y1]` | `[2, 3]` |
| **只有点** | `PromptPoint` (1) | 1 | `[x,y]` | `[1]` |
| **框 + 点** | —（见 §5 说明） | 3 | `[x0,y0, x1,y1, px,py]` | `[2, 3, 1]` |

> 坐标必须按 `x * 1024 / image.cols`、`y * 1024 / image.rows` 缩放到 **0–1024**。
> 这一步由 `creatPointInput()` 内部完成，调用方传**原图像素坐标**即可。

---

## 3. 代码示例

### 3.1 只有框

```cpp
auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();
sam2.initialize(engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::SingleTrack, 0);

std::vector<TrackerBySAM2::ParamsSam2> parms;
// type = PromptBox，使用 prompt_box
parms.push_back({TrackerBySAM2::PROMPTTYPE::PromptBox,
                 cv::Rect(100, 200, 150, 300),   // x, y, w, h
                 cv::Point(0, 0)});              // 框模式下 prompt_point 不使用
sam2.setparms(parms);

sam2.inference(frame);
cv::Rect bbox = sam2.LastRect;
```

### 3.2 只有点

```cpp
std::vector<TrackerBySAM2::ParamsSam2> parms;
// type = PromptPoint，使用 prompt_point（前景点）
parms.push_back({TrackerBySAM2::PROMPTTYPE::PromptPoint,
                 cv::Rect(),                     // 点模式下 prompt_box 不使用
                 cv::Point(960, 540)});          // 点击处（原图像素坐标）
sam2.setparms(parms);

sam2.inference(frame);
cv::Rect bbox  = sam2.LastRect;
const auto& masks = sam2.getLastMasks();         // 二值mask（0/255）
```

### 3.3 多目标（全框 或 全点）

```cpp
std::vector<TrackerBySAM2::ParamsSam2> parms;
// 全点：每个目标一个点
for (const auto& p : points) {
    parms.push_back({TrackerBySAM2::PROMPTTYPE::PromptPoint, cv::Rect(), p});
}
// 或 全框：
// for (const auto& b : boxes) {
//     parms.push_back({TrackerBySAM2::PROMPTTYPE::PromptBox, b, cv::Point(0, 0)});
// }

sam2.initialize(engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::MultiTrack, 0);
sam2.setparms(parms);            // batch = parms.size()

for (auto& frame : frames) {
    sam2.inference(frame);
    const auto& masks = sam2.getLastMasks();   // size = 目标数量
}
```

---

## 4. 约束（重要）

1. **同一批的提示类型必须一致。**
   `point_coords` 是稠密张量 `[B, N, 2]`，`N` 全批唯一。因此不能「这一行框、那一行点」。
   → **要么全框（N=2），要么全点（N=1）。**
   违反时 `setparms()` 会抛 `std::runtime_error("mixed prompt types in one batch is not supported")`。

2. **单点必须真送 N=1，不能用 `-1` 凑数。**
   实测：把单点补成 2 点（第 2 点 `label=-1`）会让模型多算一个 token，输出不等价（mask 差异很大）。

3. **非第 0 帧的提示由内部处理，调用方无需关心。**
   只有第 0 帧（condition frame）使用提示，之后靠记忆传播；后续帧由 `creatPointInput()` 自动填入
   `num_points` 个「坐标全 0 + 标签 -1」的占位点来关闭提示。

---

## 5. 关于 `label = -1`

- 语义：**「这不是一个点」（no-op / 关闭提示）**，模型内部走 `not_a_point_embed`。
- 合法用途：**非第 0 帧**关闭提示（内部逻辑），或**框+点混合时**把点数补齐到统一 N。
- **禁止**：用它把单点凑成 2 点（见 §4.2）。

### 框 + 点混合

若确实要「框 + 点」混合，需要把全批统一到 `N=3`（框 2 点 + 点 1 点）：

| 目标 | `point_coords` | `point_labels` |
|---|---|---|
| 框+点 | `[x0,y0, x1,y1, px,py]` | `[2, 3, 1]` |
| 只有框（需补齐） | `[x0,y0, x1,y1, 0,0]` | `[2, 3, -1]` |
| 只有点（需补齐） | `[0,0, 0,0, px,py]` | `[-1, -1, 1]` |

> ⚠️ 当前版本实现的是「全框」或「全点」两种模式，**未开放 N=3 的混合模式**；
> 且当前 engine 的 profile 上限为 `N=2`，若要做混合需把 engine 的 `maxShapes` 提到 `N=3` 并重建。
> 补齐点（label=-1）会引入额外的 `not_a_point` token，对结果有影响，需自行评估。

---

## 6. 对外 API 一览

| API | 说明 |
|---|---|
| `TrackerBySAM2::PROMPTTYPE::PromptBox / PromptPoint` | 提示类型枚举（0 / 1） |
| `TrackerBySAM2::ParamsSam2{type, prompt_box, prompt_point}` | 单个目标的提示参数 |
| `int setparms(std::vector<ParamsSam2>&)` | 设置全部目标提示；推导 `num_points`；分配 batch 内存 |
| `bool initialize(paths, trackType, gpuId)` | 构建引擎；`trackType` 取 `SingleTrack` / `MultiTrack` |
| `bool inference(cv::Mat& image)` | 单帧推理 |
| `cv::Rect LastRect` | 第一个目标的输出包围盒（点模式同样可用） |
| `const std::vector<cv::Mat>& getLastMasks()` | 每个目标的二值 mask（`CV_8UC1`，0/255），长度 = 目标数 |

> 注意：`LastRect` 在多目标下仍是单值（与改动前一致）；多目标请用 `getLastMasks()`。

---

## 7. 命令行用法

```bash
# 单图，框提示
./bin/SAM2 image <image> <x> <y> <w> <h>

# 视频，单目标，框提示
./bin/SAM2 video <video> <x> <y> <w> <h> [max_frames]

# 视频，多目标，框提示
./bin/SAM2 mot   <video> <x1> <y1> <w1> <h1> [x2 y2 w2 h2 ...] [max_frames]

# 视频，单目标，点提示（新增）
./bin/SAM2 point <video> <x> <y> [max_frames]
```

示例：

```bash
./bin/SAM2 point test/nanwang_third.mp4 960 540 50
```

---

## 8. 引擎要求

点模式要求 `image_decoder.engine` 的 `point_coords` 维度允许 `N=1`：

```bash
./trtexec --onnx=image_decoder.onnx --saveEngine=image_decoder.engine --fp16 \
  --minShapes=point_coords:1x1x2,point_labels:1x1,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128 \
  --optShapes=point_coords:4x2x2,point_labels:4x2,image_embed:4x256x64x64,high_res_feats_0:4x32x256x256,high_res_feats_1:4x64x128x128 \
  --maxShapes=point_coords:10x2x2,point_labels:10x2,image_embed:10x256x64x64,high_res_feats_0:10x32x256x256,high_res_feats_1:10x64x128x128
```

> 用旧的静态 `[1,2,2]` 引擎跑点模式会失败（N=1 不在 profile 范围内）。
