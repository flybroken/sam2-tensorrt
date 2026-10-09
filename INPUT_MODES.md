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
- **点** = 用「点 + 标签」表示（1=前景，0=背景）

其中 `B` = 目标数量（batch），`N` = 每个目标的点数。

---

## 2. 四种输入方式

| 输入方式 | `type` | N | `point_coords` | `point_labels` |
|---|---|---|---|---|
| **只有框** | `PROMPT_BOX` | 2 | `[x0,y0, x1,y1]` | `[2, 3]` |
| **单点** | `PROMPT_POINT` | 1 | `[x,y]` | `[1]` |
| **多点（同目标）** | `PROMPT_POINT` | K | `[x1,y1 ... xK,yK]` | `[l1 ... lK]`（1=正/0=负）|
| **框 + 点** | —（见 §5） | 3 | `[x0,y0, x1,y1, px,py]` | `[2, 3, l]` |

> 坐标由 `creatPointInput()` 内部按 `x*1024/cols, y*1024/rows` 缩放，调用方传**原图像素坐标**。

---

## 3. 代码示例

### 3.1 只有框

```cpp
auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();
sam2.initialize(engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::SingleTrack, 0);

std::vector<TrackerBySAM2::ParamsSam2> parms;
parms.push_back({TrackerBySAM2::PROMPTTYPE::PROMPT_BOX,
                 cv::Rect(100, 200, 150, 300),   // x, y, w, h
                 {}});                           // 框模式下 points 不用
sam2.setparms(parms);

sam2.inference(frame);
cv::Rect bbox = sam2.LastRect;
```

### 3.2 单点

```cpp
std::vector<TrackerBySAM2::ParamsSam2> parms;
parms.push_back({TrackerBySAM2::PROMPTTYPE::PROMPT_POINT,
                 cv::Rect(),                             // 点模式下 prompt_box 不用
                 {{cv::Point(960, 540), 1}}});            // {点, 标签}，1=正
sam2.setparms(parms);

sam2.inference(frame);
cv::Rect bbox  = sam2.LastRect;
const auto& masks = sam2.getLastMasks();                 // 二值mask（0/255）
```

### 3.3 同一目标多点（含正/负点）

```cpp
// 点 3 下圈出目标：前两个是正点，第三个是负点（排除误检区域）
std::vector<TrackerBySAM2::PromptPoint> pts = {
    {cv::Point(500, 400), 1},   // 正（前景）
    {cv::Point(520, 410), 1},   // 正
    {cv::Point(300, 300), 0},   // 负（背景）
};
std::vector<TrackerBySAM2::ParamsSam2> parms;
parms.push_back({TrackerBySAM2::PROMPTTYPE::PROMPT_POINT, cv::Rect(), pts});
sam2.setparms(parms);
```

### 3.4 多目标（全框 或 全点）

```cpp
std::vector<TrackerBySAM2::ParamsSam2> parms;
// 全点：每个目标一个点
for (const auto& p : points) {
    parms.push_back({TrackerBySAM2::PROMPTTYPE::PROMPT_POINT, cv::Rect(), {{p, 1}}});
}
// 或 全框：
// for (const auto& b : boxes) {
//     parms.push_back({TrackerBySAM2::PROMPTTYPE::PROMPT_BOX, b, {}});
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

1. **同一批的提示类型必须一致**（要么全框，要么全点）。
   违反时 `setparms()` 抛 `std::runtime_error("mixed prompt types in one batch is not supported")`。

2. **同一批的点数必须一致**（`point_coords` 是 `[B,N,2]` 稠密张量，N 全批唯一）。
   违反时抛 `std::runtime_error("all targets in one batch must use the same number of points")`。
   > 不支持用 `-1` 补齐：实测会多出 `not_a_point` token，误差随占位符数量线性增长。

3. **每个点必须带标签**（1=正，0=负），且**至少 1 个正点**。
   全负会让 mask 面积为 0（模型硬约束）。违反时抛
   `std::runtime_error("point prompt requires at least one positive point (label=1)")`。

4. **点数上限 8**（`MAX_POINTS`）。超过时**截断取前 8 个**（不报错）。

5. **点的顺序无关**（排列不变，实测差异 2.7e-05）。

6. **非第 0 帧的提示由内部处理**，调用方无需关心；只有第 0 帧用提示，之后靠记忆传播。

---

## 5. 关于 `label` 取值

| 值 | 含义 | 允许用户使用 |
|---|---|---|
| `1` | 正点（前景） | ✅ |
| `0` | 负点（背景） | ✅ |
| `2` / `3` | 框的左上/右下角点 | 内部使用（框模式） |
| `-1` | **「这不是一个点」占位**（no-op） | ❌ **禁止** |

⚠️ **`-1` 与 `0` 语义完全不同**：`-1` 是"关掉提示"（模型自行推断），`0` 是"否定此区域"。
实测：单点 `label=-1` 面积 33265，`label=0` 面积 **0**。
`-1` 仅由内部在**非第 0 帧**使用，用户侧不得传入。

### 框 + 点混合

若确实要「框 + 点」，需全批统一到 `N=3`、labels `[2,3,l]`。
> ⚠️ 当前版本实现的是「全框」或「全点」两种模式，**未开放 N=3 混合模式**；
> 且 engine 的点数上限为 8，`N=3` 在范围内，但混合模式的构造逻辑需自行扩展。

---

## 6. 对外 API 一览

| API | 说明 |
|---|---|
| `PROMPTTYPE::PROMPT_BOX / PROMPT_POINT` | 提示类型枚举（0 / 1） |
| `TrackerBySAM2::PromptPoint{point, label}` | 单个提示点（点 + 强制标签） |
| `TrackerBySAM2::ParamsSam2{type, prompt_box, points}` | 单个目标的提示参数 |
| `int setparms(std::vector<ParamsSam2>&)` | 设置全部目标提示；推导 `num_points`；分配 batch 内存 |
| `bool initialize(paths, trackType, gpuId)` | 构建引擎；`trackType` 取 `SingleTrack` / `MultiTrack` |
| `bool inference(cv::Mat& image)` | 单帧推理 |
| `cv::Rect LastRect` | 第一个目标的输出包围盒 |
| `const std::vector<cv::Mat>& getLastMasks()` | 每个目标的二值 mask（`CV_8UC1`，0/255），长度 = 目标数 |
| `MAX_POINTS` | 每目标点数上限常量（= 8） |

> ⚠️ `setparms()` **不会重置** `current_frame`。同一实例若要在中途更换提示，
> 需先经过 `sam2Process()`（它会重置记忆状态与帧计数），否则提示会被当作非首帧而忽略。

---

## 7. 命令行用法

```bash
# 单图，框提示
./bin/SAM2 image <image> <x> <y> <w> <h>

# 视频，单目标，框提示
./bin/SAM2 video <video> <x> <y> <w> <h> [max_frames]

# 视频，多目标，框提示
./bin/SAM2 mot   <video> <x1> <y1> <w1> <h1> [x2 y2 w2 h2 ...] [max_frames]

# 视频，单目标，点提示（支持多点：每 3 个参数 = x y label）
./bin/SAM2 point <video> <x> <y> <label> [x2 y2 l2 ...] [max_frames]
```

示例：

```bash
# 单点
./bin/SAM2 point test/nanwang_third.mp4 960 540 1 50

# 同一目标 3 点：2 正 1 负
./bin/SAM2 point test/nanwang_third.mp4 960 540 1 970 550 1 300 300 0 50
```

---

## 8. 引擎要求

点数上限由 engine 的 `maxShapes` 决定（当前为 8）：

```bash
./trtexec --onnx=image_decoder.onnx --saveEngine=image_decoder.engine --fp16 \
  --minShapes=point_coords:1x1x2,point_labels:1x1,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128 \
  --optShapes=point_coords:4x2x2,point_labels:4x2,image_embed:4x256x64x64,high_res_feats_0:4x32x256x256,high_res_feats_1:4x64x128x128 \
  --maxShapes=point_coords:10x8x2,point_labels:10x8,image_embed:10x256x64x64,high_res_feats_0:10x32x256x256,high_res_feats_1:10x64x128x128
```

> - `optShapes` 的 N=2 表示"框/单点"为主路径。
> - 若要修改点数上限，需同步改 `MAX_POINTS` 并重建引擎。
