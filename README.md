# 基于并行加速的语义级 3D 场景重建与 VLM 决策系统

本项目是课程选题“基于并行加速的语义级 3D 场景重建与视觉语言（VLM）决策系统”的无 API 版本实现。当前目标是先跑通本地视觉闭环：

```text
视频输入 -> Depth-Anything-V2 深度估计 -> 相对深度转 Pseudo-LiDAR 伪距离
       -> YOLO-World 语义检测 -> Numba CUDA 点云反投影
       -> OpenCV/Open3D 可视化 -> 本地规则决策 -> 性能报告
```

当前版本不依赖云端 VLM API。决策层默认使用本地规则输出中文风险建议，后续拿到 Qwen-VL/DashScope API key 后，可以在 `src/decision.py` 中扩展云端 VLM 客户端。

## 当前能力

- 单目视频逐帧深度估计：默认使用 Depth-Anything-V2 Small `vits`。
- 语义目标检测：默认使用 Ultralytics YOLO-World `yolov8s-worldv2.pt`。
- 点云反投影：提供纯 Python 串行 CPU、NumPy CPU、Numba CUDA 三种实现。
- 并行性能量化：输出 CPU 串行到 CUDA 的耗时、FPS 和加速比。
- 可视化输出：生成原视频、深度热力图、语义叠加图、点云俯视图组成的 2x2 演示视频。
- 3D 文件输出：关键帧点云保存为 Open3D 可打开的 `.ply` 文件。
- 无 API 决策：根据最近语义目标、中心通道占比和伪距离输出本地避障建议。

## 目录结构

```text
.
├── configs/
│   └── demo.yaml                 # 主配置文件
├── data/
│   └── input/                    # 默认演示视频目录
├── docs/
│   └── 使用说明.md               # 简短运行说明
├── models/                       # Depth-Anything-V2 权重
├── outputs/                      # run_demo 输出
├── benchmarks/                   # benchmark 输出
├── src/
│   ├── prepare_assets.py          # 下载/准备模型和演示素材
│   ├── run_demo.py                # 运行完整本地视觉闭环
│   ├── benchmark.py               # CPU/GPU 点云反投影性能对比
│   ├── decide.py                  # 单独运行本地决策
│   ├── config.py                  # 配置读取与路径解析
│   ├── depth_estimator.py         # Depth-Anything-V2 和 fallback 深度
│   ├── semantic_detector.py       # YOLO-World 语义检测与颜色映射
│   ├── projector.py               # CPU/NumPy/CUDA 反投影核心
│   ├── visualization.py           # 深度图、语义图、点云预览与 PLY 输出
│   ├── decision.py                # 本地规则决策与云端 VLM 占位接口
│   └── utils.py                   # 下载、视频、CSV、图像工具函数
├── tests/
│   └── test_projector.py          # 投影器单元测试
└── third_party/
    └── Depth-Anything-V2/         # 官方仓库，由 prepare_assets 克隆
```

## 环境

本机当前使用 `alg` conda 环境，已验证：

- Python `3.10.19`
- NVIDIA RTX 4060 Laptop GPU 8GB
- CUDA Toolkit `13.0`
- PyTorch `2.9.1+cu130`
- Numba CUDA 可用
- Open3D 可用
- Ultralytics YOLO-World 可用

新开终端时先进入环境：

```bash
conda activate alg
cd /home/cells/ai
```

如果换到一台新机器，至少需要安装：

```bash
python -m pip install -U numba open3d transformers opencv-python ultralytics matplotlib pyyaml requests
```

注意：PyTorch CUDA 版本最好按机器 CUDA/驱动重新安装，不建议盲目复制 pip 命令。

## 快速开始

第一次运行先准备模型和素材：

```bash
python -m src.prepare_assets
```

该命令会做几件事：

- 克隆 Depth-Anything-V2 官方仓库到 `third_party/Depth-Anything-V2`
- 下载 `depth_anything_v2_vits.pth` 到 `models/`
- 下载 OpenCV 官方 `vtest.avi` 并转成默认 `data/input/demo.mp4`
- 预热 YOLO-World 权重下载

运行完整演示：

```bash
python -m src.run_demo --video data/input/demo.mp4
```

快速验收可以限制帧数：

```bash
python -m src.run_demo --video data/input/demo.mp4 --max-frames 5
```

运行 CPU/GPU 反投影基准测试：

```bash
python -m src.benchmark --video data/input/demo.mp4 --max-frames 60
```

查看本地规则决策：

```bash
python -m src.decide --snapshot outputs/keyframes/frame_0000.jpg
```

## 输出文件

`run_demo` 会生成：

- `outputs/demo_side_by_side.mp4`：2x2 演示视频。
- `outputs/performance.csv`：逐帧深度估计、语义检测、CUDA 投影耗时。
- `outputs/keyframes/`：关键帧、深度图、语义图、点云预览。
- `outputs/pointclouds/*.ply`：Open3D 点云文件。
- `outputs/decision/latest_scene_metrics.json`：本地决策使用的结构化指标。
- `outputs/decision/decision_report.txt`：中文风险等级与避障建议。
- `outputs/run_summary.md`：本次运行摘要。

`benchmark` 会生成：

- `benchmarks/results.csv`：`cpu_loop_ms`、`cpu_numpy_ms`、`gpu_ms`、`speedup`、点数等。

当前一次短测试中，串行 CPU 到 CUDA 的平均加速比约为 `12.41x`。NumPy CPU 通常比 CUDA 更快，是因为 NumPy 已经调用底层 C 向量化实现；课程报告里建议把“纯 Python 串行 CPU -> CUDA”作为并行加速对比，把 NumPy 作为工程参考。

## 配置说明

主配置位于 `configs/demo.yaml`。

常用字段：

- `runtime.max_frames`：默认处理帧数。
- `runtime.process_width`：视频缩放宽度，降低显存和渲染压力。
- `runtime.use_yolo`：是否启用 YOLO-World。
- `projection.stride`：点云采样步长，越小点越密，计算和渲染越慢。
- `projection.fov_degrees`：没有相机标定时用于反推内参的水平视场角。
- `projection.pseudo_min_z` / `projection.pseudo_max_z`：相对深度映射到伪距离的范围。
- `decision.danger_z` / `decision.warning_z`：本地规则决策阈值。

建议调参顺序：

1. 先调 `runtime.process_width` 和 `projection.stride`，让演示速度稳定。
2. 再调 `model.yolo_conf`，减少误检或漏检。
3. 最后根据画面效果调 `projection.invert_depth` 和伪距离范围。

## 核心数据流

### 1. 深度估计

入口：`src/depth_estimator.py`

`DepthAnythingV2Estimator.predict(frame_bgr)` 返回 `DepthResult`：

- `depth`：与输入帧同尺寸的相对深度图。
- `backend`：`official` 或 `fallback`。
- `elapsed_ms`：深度估计耗时。

如果官方仓库或权重缺失，且 `allow_fallback_depth=true`，程序会使用 `FallbackDepthEstimator` 生成合成深度，保证 CUDA 投影和可视化仍然可测。

### 2. 相对深度转伪距离

入口：`src/utils.py`

`relative_depth_to_pseudo_z` 将 Depth-Anything-V2 输出归一化到 `[pseudo_min_z, pseudo_max_z]`。

重要：当前 `Z` 是 Pseudo-LiDAR 伪距离，不是真实米制深度。没有相机内参、尺度标定或已知距离参照时，不要在报告中写成真实距离。

### 3. 语义检测

入口：`src/semantic_detector.py`

`SemanticDetector.predict(frame_bgr)` 返回：

- 检测框列表 `detections`
- 与图像同尺寸的 `label_mask`
- 推理耗时

当前采用检测框区域给点云赋标签，不是像素级分割。这样实现轻量稳定，适合课程演示；后续可以替换为 SAM2 或实例分割模型。

### 4. CUDA 点云反投影

入口：`src/projector.py`

核心公式：

```text
Z = depth(u, v)
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

实现包括：

- `project_cpu_loop`：纯 Python 串行 baseline，用于展示 CUDA 加速。
- `project_cpu`：NumPy 向量化实现，用于工程参考。
- `project_cuda`：Numba CUDA kernel，每个线程负责一个采样像素。

如果要修改 CUDA kernel，优先看 `_back_project_kernel` 中关于线程网格、深度过滤、语义颜色和 BGR/RGB 转换的中文注释。

### 5. 可视化

入口：`src/visualization.py`

主要输出：

- 深度热力图：`depth_to_heatmap`
- 语义叠加图：`semantic_overlay`
- 点云俯视图：`render_topdown`
- PLY 点云：`save_point_cloud_ply`
- 关键帧 3D 散点图：`save_point_cloud_preview`

视频里的点云图是快速俯视投影，适合逐帧输出；`.ply` 文件才是可在 Open3D/CloudCompare 中查看的 3D 点云。

### 6. 决策层

入口：`src/decision.py`

当前使用 `LocalRuleDecisionClient`：

- 读取最近目标的伪距离。
- 判断目标是否位于画面中心通道。
- 根据 `danger_z` 和 `warning_z` 输出中文风险等级和建议。

`CloudVLMDecisionClient` 是云端 VLM 占位类。后续接入 Qwen-VL 时建议保持 `DecisionClient.analyze(metrics, snapshot_path)` 这个接口不变。

## 后续开发建议

### 接入真实 Qwen-VL

推荐做法：

1. 设置环境变量：

```bash
export DASHSCOPE_API_KEY="你的 key"
```

2. 在 `CloudVLMDecisionClient.analyze` 中读取：

- `snapshot_path`：关键帧演示图。
- `metrics`：最近目标、中心通道、伪距离、检测列表等结构化信息。

3. Prompt 中明确说明“深度为相对伪距离，不是米制真值”。

4. 在配置中增加 `decision.backend: local|qwen`，由 `run_demo` 根据配置选择客户端。

### 替换演示视频

把自己的视频放到：

```text
data/input/demo.mp4
```

或运行时指定：

```bash
python -m src.run_demo --video /path/to/your_video.mp4
```

建议视频：

- 30-90 秒。
- 720P 或 1080P。
- 画面稳定。
- 有明显前后距离关系。
- 包含行人、车辆、障碍物、道路或通道。

### 使用真实相机内参

当前 `build_intrinsics` 通过 FOV 估算内参。如果后续有相机标定结果，可以：

1. 在 `configs/demo.yaml` 增加 `fx/fy/cx/cy`。
2. 修改 `build_intrinsics` 或新增 `load_intrinsics`。
3. 在报告中区分“标定内参”和“伪距离尺度”。

只有内参还不够得到真实米制距离；还需要深度尺度标定或 metric depth 模型。

### 加入像素级分割

当前语义标签来自 YOLO 检测框。若要提高点云语义精度，可以：

- 用 YOLO segmentation 模型替换 YOLO-World。
- 接入 SAM2，根据检测框生成 mask。
- 保持输出为 `label_mask`，这样 `PointCloudProjector` 不需要改。

### 提高实时性

优先尝试：

- 增大 `projection.stride`。
- 降低 `runtime.process_width`。
- 降低 `model.depth_input_size`。
- 减少 YOLO 检测类别。
- 每 N 帧跑一次 YOLO，其他帧复用上一帧标签。

进阶加分项：

- CUDA Streams：让深度推理和点云投影异步流水。
- TensorRT：加速 Depth-Anything-V2。
- 多帧点云累积：形成局部地图。

## 测试与验收

语法检查：

```bash
python -m py_compile src/*.py
```

单元测试：

```bash
python -m unittest discover -s tests
```

短演示：

```bash
python -m src.run_demo --video data/input/demo.mp4 --max-frames 5
```

基准测试：

```bash
python -m src.benchmark --video data/input/demo.mp4 --max-frames 20
```

验收重点：

- `outputs/demo_side_by_side.mp4` 能正常打开。
- `outputs/performance.csv` 有逐帧耗时。
- `benchmarks/results.csv` 有 CPU/GPU 对比。
- `outputs/pointclouds/*.ply` 非空。
- `outputs/decision/decision_report.txt` 有中文风险建议。

## 常见问题

### `xFormers not available`

Depth-Anything-V2 的可选加速库提示，不影响运行。当前 PyTorch CUDA 推理可用即可。

### Numba 提示 GPU under-utilization

短测试或较大 `stride` 会让采样点数较少，CUDA 网格很小，因此会提示 GPU 利用率不足。完整高分辨率视频或更小 stride 下会改善。

### CUDA benchmark 比 NumPy 慢

这是正常现象。NumPy 已经是底层 C 向量化实现，而 `project_cuda` 的计时包含 CPU/GPU 拷贝。课程并行对比建议使用 `cpu_loop_ms` 对比 `gpu_ms`，同时在报告中说明 NumPy 是工程优化参考。

### YOLO-World 第一次运行很慢

第一次会下载 YOLO-World 权重和 CLIP 相关权重。下载完成后后续运行会快很多。

### 没有 API key 怎么办

当前版本不需要 API key。本地规则决策已经能生成中文风险建议。拿到 API key 后再扩展 `CloudVLMDecisionClient`。

### 输出深度能不能当真实距离

不能。当前是相对深度映射的伪距离，只适合做 Pseudo-LiDAR 演示和相对风险判断。如果要真实距离，需要相机标定、尺度校准或 metric depth 模型。

## 参考

- Depth-Anything-V2: https://github.com/DepthAnything/Depth-Anything-V2
- Ultralytics YOLO-World: https://docs.ultralytics.com/models/yolo-world/
- Numba CUDA: https://numba.readthedocs.io/en/stable/cuda/index.html
- OpenCV sample video: https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi
