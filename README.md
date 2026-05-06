# 基于并行加速的语义级 3D 场景重建与 VLM 决策系统

本项目是课程选题“基于并行加速的语义级 3D 场景重建与视觉语言（VLM）决策系统”的无 API 版本实现。当前目标是先跑通本地视觉闭环：

```text
视频输入 -> Depth-Anything-V2 深度估计 -> 相对深度转 Pseudo-LiDAR 伪距离
       -> SAM 3 像素级语义分割 -> Numba CUDA 点云反投影
       -> OpenCV/Open3D 可视化 -> 本地规则决策 -> 性能报告
```

当前版本不依赖云端 VLM API。决策层默认使用本地规则输出中文风险建议，也已支持本地 Qwen3-VL 对关键帧和结构化指标做视觉语言决策分析。

## 当前能力

- 单目视频逐帧深度估计：默认使用 Depth-Anything-V2 Large `vitl` 高精度权重。
- 语义分割：默认使用 Ultralytics SAM 3 和本地 `third_party/sam3/sam3.pt` 权重输出像素级 mask。
- 点云反投影：提供纯 Python 串行 CPU、NumPy CPU、Numba CUDA 三种实现。
- 进阶深度滤波：提供 CUDA 边缘保持邻域统计滤波，减少点云毛刺。
- 并行性能量化：输出 CPU 串行到 CUDA 的耗时、FPS 和加速比。
- 可视化输出：生成原视频、深度热力图、语义叠加图、点云俯视图组成的 2x2 演示视频。
- 3D 文件输出：关键帧点云保存为 Open3D 可打开的 `.ply` 文件。
- 交互查看：支持 Open3D 打开 PLY，也可配置实时 Open3D 点云窗口。
- 无 API 决策：根据最近语义目标、中心通道占比和伪距离输出本地避障建议。
- 本地 VLM：在 8GB 显存 RTX 4060 Laptop 上默认部署 `Qwen/Qwen3-VL-2B-Instruct`。
- 自动自评：每次演示/基准测试结束后生成效果评估和改进建议。

## 目录结构

```text
.
├── configs/
│   └── demo.yaml                 # 主配置文件
├── data/
│   └── input/                    # 默认演示视频目录
├── docs/
│   ├── 使用说明.md               # 简短运行说明
│   └── 效果标准与自评.md         # 好效果标准和自评逻辑
├── models/                       # Depth-Anything-V2 权重
├── outputs/                      # run_demo 输出
├── benchmarks/                   # benchmark 输出
├── src/
│   ├── prepare_assets.py          # 下载/准备模型和演示素材
│   ├── prepare_qwen3_vl.py        # 下载/准备本地 Qwen3-VL 权重
│   ├── run_demo.py                # 运行完整本地视觉闭环
│   ├── qwen3_vl_chat.py           # 使用本地 Qwen3-VL 分析关键帧
│   ├── qwen3_vl_local.py          # 本地 Qwen3-VL 推理封装
│   ├── ui_app.py                  # 浏览器实时决策 UI
│   ├── benchmark.py               # CPU/GPU 点云反投影性能对比
│   ├── decide.py                  # 单独运行本地决策
│   ├── config.py                  # 配置读取与路径解析
│   ├── depth_estimator.py         # Depth-Anything-V2 和 fallback 深度
│   ├── filters.py                 # CUDA 深度统计滤波
│   ├── semantic_detector.py       # SAM 3 语义分割与颜色映射
│   ├── projector.py               # CPU/NumPy/CUDA 反投影核心
│   ├── visualization.py           # 深度图、语义图、点云预览与 PLY 输出
│   ├── live_viewer.py             # 可选 Open3D 实时窗口
│   ├── view_pointcloud.py         # 打开 PLY 点云进行交互式查看
│   ├── evaluation.py              # 每次运行后的自动自评报告
│   ├── decision.py                # 本地规则决策与云端 VLM 占位接口
│   └── utils.py                   # 下载、视频、CSV、图像工具函数
├── tests/
│   └── test_projector.py          # 投影器单元测试
└── third_party/
    ├── Depth-Anything-V2/         # 官方仓库，由 prepare_assets 克隆
    └── sam3/                      # SAM 3 本地权重目录，默认读取 sam3.pt
```

## 环境

本机当前使用 `alg` conda 环境，已验证：

- Python `3.10.19`
- NVIDIA RTX 4060 Laptop GPU 8GB
- CUDA Toolkit `13.0`
- PyTorch `2.9.1+cu130`
- Numba CUDA 可用
- Open3D 可用
- Ultralytics SAM 3 可用（需要 `ultralytics>=8.3.237`）

新开终端时先进入环境：

```bash
conda activate alg
cd /home/cells/ai
```

如果换到一台新机器，至少需要安装：

```bash
python -m pip install -U numba open3d transformers accelerate opencv-python "ultralytics>=8.3.237" timm matplotlib pyyaml requests modelscope qwen-vl-utils sentencepiece protobuf
```

注意：PyTorch CUDA 版本最好按机器 CUDA/驱动重新安装，不建议盲目复制 pip 命令。

## 快速开始

第一次运行先准备模型和素材：

```bash
python -m src.prepare_assets
```

该命令会做几件事：

- 克隆 Depth-Anything-V2 官方仓库到 `third_party/Depth-Anything-V2`
- 下载 `depth_anything_v2_vitl.pth` 到 `models/`
- 下载 OpenCV 官方 `vtest.avi` 并转成默认 `data/input/demo.mp4`
- 检查 `third_party/sam3/sam3.pt` 和 Ultralytics SAM 3 接口是否可用

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

下载并运行本地 Qwen3-VL：

```bash
python -m src.prepare_qwen3_vl --source modelscope
python -m src.qwen3_vl_chat --image outputs/keyframes/frame_0090.jpg --metrics outputs/decision/latest_scene_metrics.json
```

启动浏览器实时决策 UI：

```bash
python -m src.ui_app --host 127.0.0.1 --port 7860
```

UI 默认使用本地 `Qwen/Qwen3-VL-2B-Instruct` 做右侧决策，并把左侧视频放慢到每帧约 `250ms`、每 `30` 帧更新一次 VLM 决策，避免 8GB 显存下频繁推理卡死。需要切回原来的 Depth/SAM3 + 本地规则链路时，可在页面右上角切换为“本地规则”，或用命令行启动：

```bash
python -m src.ui_app --decision-backend local_rules --decision-stride 1 --frame-delay-ms 0
```

打开最近生成的 PLY 点云：

```bash
python -m src.view_pointcloud
```

## 输出文件

`run_demo` 会生成：

- `outputs/demo_side_by_side.mp4`：2x2 演示视频。
- `outputs/performance.csv`：逐帧深度估计、语义分割、CUDA 投影耗时。
- `outputs/keyframes/`：关键帧、深度图、语义图、点云预览。
- `outputs/pointclouds/*.ply`：Open3D 点云文件。
- `outputs/decision/latest_scene_metrics.json`：本地决策使用的结构化指标。
- `outputs/decision/decision_report.txt`：中文风险等级与避障建议。
- `outputs/decision/qwen3_vl_report.txt`：本地 Qwen3-VL 生成的关键帧视觉语言分析。
- `outputs/run_summary.md`：本次运行摘要。
- `outputs/self_evaluation.md`：本次演示效果自评和下一步改进建议。

`benchmark` 会生成：

- `benchmarks/results.csv`：`cpu_loop_ms`、`cpu_numpy_ms`、`gpu_ms`、`speedup`、点数等。
- `benchmarks/self_evaluation.md`：基准测试自评。

当前一次短测试中，串行 CPU 到 CUDA 的平均加速比约为 `12.41x`。NumPy CPU 通常比 CUDA 更快，是因为 NumPy 已经调用底层 C 向量化实现；课程报告里建议把“纯 Python 串行 CPU -> CUDA”作为并行加速对比，把 NumPy 作为工程参考。

## 配置说明

主配置位于 `configs/demo.yaml`。

常用字段：

- `runtime.max_frames`：默认处理帧数。
- `runtime.process_width`：视频缩放宽度，降低显存和渲染压力。
- `runtime.use_sam3`：是否启用 SAM 3 语义分割。
- `projection.stride`：点云采样步长，越小点越密，计算和渲染越慢。
- `projection.fov_degrees`：没有相机标定时用于反推内参的水平视场角。
- `projection.pseudo_min_z` / `projection.pseudo_max_z`：相对深度映射到伪距离的范围。
- `filtering.enabled`：是否启用 CUDA 深度统计滤波。
- `filtering.depth_jump_threshold`：邻域像素距离差阈值，越小越保边，越大越平滑。
- `visualization.live_open3d`：是否开启实时 Open3D 交互窗口，远程终端建议保持 `false`。
- `evaluation.enabled`：是否在每次运行后自动生成自评报告。
- `decision.danger_z` / `decision.warning_z`：本地规则决策阈值。
- `vlm.qwen3_vl_model_id`：本地 Qwen3-VL 模型 ID，当前默认 `Qwen/Qwen3-VL-2B-Instruct`。
- `vlm.qwen3_vl_local_dir`：本地 Qwen3-VL 权重目录。
- `vlm.qwen3_vl_max_memory_gb`：加载 Qwen3-VL 时给 GPU 的显存预算，8GB 显卡建议保留在 `6.0` 左右。

建议调参顺序：

1. 先调 `runtime.process_width` 和 `projection.stride`，让演示速度稳定。
2. 再调 `model.sam3_conf`，减少误分割或漏分割。
3. 然后调 `filtering.depth_jump_threshold`，让点云更平滑但不过度糊掉边缘。
4. 最后根据画面效果调 `projection.invert_depth` 和伪距离范围。

## 核心数据流

### 本地 Qwen3-VL 部署

入口：`src/prepare_qwen3_vl.py`、`src/qwen3_vl_chat.py`

本机硬件是 NVIDIA RTX 4060 Laptop GPU 8GB，因此默认选择 `Qwen/Qwen3-VL-2B-Instruct`。2B 权重约 4GB，配合 `device_map="auto"` 和 6GB GPU 显存预算可以在本机完成图像问答；4B/8B 版本建议优先考虑量化、CPU offload 或更大显存后再测试。

依赖已安装在 `alg` 环境中：

```bash
python -m pip install -U transformers accelerate modelscope qwen-vl-utils sentencepiece protobuf
```

准备权重：

```bash
python -m src.prepare_qwen3_vl --source modelscope
```

对关键帧生成 VLM 决策报告：

```bash
python -m src.qwen3_vl_chat \
  --image outputs/keyframes/frame_0090.jpg \
  --metrics outputs/decision/latest_scene_metrics.json \
  --max-new-tokens 128
```

输出文本保存到 `outputs/decision/qwen3_vl_report.txt`。

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

### 3. 语义分割

入口：`src/semantic_detector.py`

`SemanticDetector.predict(frame_bgr)` 返回：

- 语义实例外接框列表 `detections`
- 与图像同尺寸的像素级 `label_mask`
- 推理耗时

当前使用 Ultralytics `SAM3SemanticPredictor`，对配置中的文本概念逐类分割，并把实例 mask 合成为点云可直接索引的标签图。

### 4. CUDA 深度统计滤波

入口：`src/filters.py`

`DepthFilter.filter_auto(z_map)` 会在 CUDA 可用时调用 `cuda_filter`。每个线程负责一个像素，只统计与中心像素差异不超过阈值的邻居，减少深度突变点，同时尽量保留障碍物边界。

这对应课程方案中的“统计学并行滤波”进阶项。

### 5. CUDA 点云反投影

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

### 6. 可视化

入口：`src/visualization.py`、`src/live_viewer.py`、`src/view_pointcloud.py`

主要输出：

- 深度热力图：`depth_to_heatmap`
- 语义叠加图：`semantic_overlay`
- 点云俯视图：`render_topdown`
- PLY 点云：`save_point_cloud_ply`
- 关键帧 3D 散点图：`save_point_cloud_preview`
- 可选实时窗口：`Open3DLiveViewer`
- 离线交互查看：`python -m src.view_pointcloud`

视频里的点云图是快速俯视投影，适合逐帧输出；`.ply` 文件才是可在 Open3D/CloudCompare 中查看的 3D 点云。

### 7. 决策层

入口：`src/decision.py`

当前使用 `LocalRuleDecisionClient`：

- 读取最近目标的伪距离。
- 判断目标是否位于画面中心通道。
- 根据 `danger_z` 和 `warning_z` 输出中文风险等级和建议。

`CloudVLMDecisionClient` 是云端 VLM 占位类。后续接入 Qwen-VL 时建议保持 `DecisionClient.analyze(metrics, snapshot_path)` 这个接口不变。

### 8. 自动自评

入口：`src/evaluation.py`

每次 `run_demo` 后会写出 `outputs/self_evaluation.md`，每次 `benchmark` 后会写出 `benchmarks/self_evaluation.md`。自评会检查视频、点云、后端、耗时、决策报告和加速比，并给出下一步改进建议。

更详细的效果标准见 [docs/效果标准与自评.md](docs/效果标准与自评.md)。

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

### 调整 SAM 3 语义分割

当前语义标签来自 SAM 3 像素级 mask。若要提高点云语义精度，可以：

- 在 `model.sam3_classes` 中把类别写成更贴合场景的英文短语。
- 降低 `model.sam3_conf` 提高召回，或升高它减少误分割。
- 保持输出为 `label_mask`，这样 `PointCloudProjector` 不需要改。

### 提高实时性

优先尝试：

- 增大 `projection.stride`。
- 降低 `runtime.process_width`。
- 降低 `model.depth_input_size`。
- 减少 SAM 3 文本概念数量。
- 每 N 帧跑一次 SAM 3，其他帧复用上一帧标签。

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

深度估计和 SAM 3 语义分割的正确性指标测试：

```bash
python -m unittest tests.test_quality_checks
```

真实 `vitl`/SAM 3 模型 smoke test 默认跳过；正式验收前可显式加载大模型检查输出契约：

```bash
RUN_HEAVY_MODEL_TESTS=1 python -m unittest tests.test_model_quality_smoke
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
- `outputs/self_evaluation.md` 对本次运行给出分数、问题和改进建议。

## 常见问题

### `xFormers not available`

Depth-Anything-V2 的可选加速库提示，不影响运行。当前 PyTorch CUDA 推理可用即可。

### Numba 提示 GPU under-utilization

短测试或较大 `stride` 会让采样点数较少，CUDA 网格很小，因此会提示 GPU 利用率不足。完整高分辨率视频或更小 stride 下会改善。

### CUDA benchmark 比 NumPy 慢

这是正常现象。NumPy 已经是底层 C 向量化实现，而 `project_cuda` 的计时包含 CPU/GPU 拷贝。课程并行对比建议使用 `cpu_loop_ms` 对比 `gpu_ms`，同时在报告中说明 NumPy 是工程优化参考。

### SAM 3 第一次运行很慢或显存占用高

SAM 3 权重约 3.3GB，首次加载和文本特征构建会比较慢。若显存紧张，优先降低 `runtime.process_width`、增大 `projection.stride`，或减少 `model.sam3_classes`。

### 没有 API key 怎么办

当前版本不需要 API key。本地规则决策已经能生成中文风险建议。拿到 API key 后再扩展 `CloudVLMDecisionClient`。

### 输出深度能不能当真实距离

不能。当前是相对深度映射的伪距离，只适合做 Pseudo-LiDAR 演示和相对风险判断。如果要真实距离，需要相机标定、尺度校准或 metric depth 模型。

## 参考

- Depth-Anything-V2: https://github.com/DepthAnything/Depth-Anything-V2
- Ultralytics SAM 3: https://docs.ultralytics.com/models/sam-3/
- Numba CUDA: https://numba.readthedocs.io/en/stable/cuda/index.html
- OpenCV sample video: https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi
