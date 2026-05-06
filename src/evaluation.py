from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2

from .config import resolve_path


@dataclass
class QualityReport:
    """一次运行的自评结果。"""

    score: int
    level: str
    strengths: list[str]
    issues: list[str]
    improvements: list[str]


def _average(rows: list[dict[str, Any]], key: str, skip_first: bool = True) -> float:
    """计算平均值；默认跳过第一帧，避免模型加载/JIT 编译时间干扰评价。"""

    values = []
    iterable = rows[1:] if skip_first and len(rows) > 1 else rows
    for row in iterable:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean(values) if values else 0.0


def _video_info(path: Path) -> tuple[bool, int, tuple[int, int]]:
    """读取视频可打开性、帧数和尺寸。"""

    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) if ok else (0, 0)
    cap.release()
    return ok, frames, size


def evaluate_demo_run(
    config: dict,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    frame_count: int,
    expected_frames: int | None = None,
) -> QualityReport:
    """评价 `run_demo` 的输出效果，并写出 Markdown 报告。

    评价维度刻意采用课程演示可观察的标准：是否使用正式模型、点云是否非空、
    语义是否有结果、CUDA 是否生效、视频/PLY/决策文件是否生成、性能是否稳定。
    这些指标不能替代人工审美检查，但能在每次运行后自动指出下一步该调哪里。
    """

    output_dir = resolve_path(output_dir)
    strengths: list[str] = []
    issues: list[str] = []
    improvements: list[str] = []
    score = 100

    video_path = output_dir / "demo_side_by_side.mp4"
    video_ok, video_frames, video_size = _video_info(video_path)
    if video_ok and video_frames > 0:
        strengths.append(f"演示视频可打开，共 {video_frames} 帧，尺寸 {video_size[0]}x{video_size[1]}。")
    else:
        score -= 25
        issues.append("演示视频不可打开或帧数为 0。")
        improvements.append("优先检查 OpenCV VideoWriter 编码器和输出路径。")

    avg_points = _average(rows, "point_count", skip_first=False)
    if avg_points >= 15000:
        strengths.append(f"平均点云数量约 {avg_points:.0f}，点云密度足够展示空间结构。")
    else:
        score -= 12
        issues.append(f"平均点云数量偏少：{avg_points:.0f}。")
        improvements.append("可降低 `projection.stride`，例如从 4 调到 3 或 2。")

    depth_backends = {str(row.get("depth_backend")) for row in rows}
    projection_backends = {str(row.get("projection_backend")) for row in rows}
    semantic_backends = {str(row.get("semantic_backend")) for row in rows}
    filter_backends = {str(row.get("filter_backend")) for row in rows if row.get("filter_backend") is not None}

    if "official" in depth_backends:
        strengths.append(f"深度估计使用官方 Depth-Anything-V2 {config['model']['depth_encoder']} 权重。")
    else:
        score -= 15
        issues.append("深度估计未使用 official 后端，可能处于 fallback 模式。")
        improvements.append(f"运行 `python -m src.prepare_assets`，确认 `{config['paths']['depth_checkpoint']}` 存在。")

    if "cuda" in projection_backends:
        strengths.append("点云反投影使用 CUDA kernel。")
    else:
        score -= 15
        issues.append("点云反投影没有使用 CUDA。")
        improvements.append("检查 `torch.cuda.is_available()` 和 `numba.cuda.is_available()`。")

    if "cuda_filter" in filter_backends:
        strengths.append("深度统计滤波使用 CUDA 并行实现。")
    elif "disabled" in filter_backends:
        score -= 5
        issues.append("深度滤波处于关闭状态。")
        improvements.append("在配置中启用 `filtering.enabled: true`。")
    elif filter_backends:
        score -= 5
        issues.append(f"深度滤波未使用 CUDA：{sorted(filter_backends)}。")
        improvements.append("检查 Numba CUDA 是否可用。")

    if "sam3" in semantic_backends:
        strengths.append("语义分割使用 SAM 3 像素级 mask，点云可按类别着色。")
    else:
        score -= 10
        issues.append("语义分割没有产生 SAM 3 输出。")
        improvements.append("确认 `third_party/sam3/sam3.pt` 存在、Ultralytics 版本为 8.3.237+，或降低 `model.sam3_conf`。")

    avg_depth_ms = _average(rows, "depth_ms")
    avg_semantic_ms = _average(rows, "semantic_ms")
    avg_filter_ms = _average(rows, "filter_ms")
    avg_projection_ms = _average(rows, "projection_ms")
    if avg_depth_ms:
        strengths.append(
            f"稳态耗时：深度 {avg_depth_ms:.1f} ms，语义 {avg_semantic_ms:.1f} ms，"
            f"滤波 {avg_filter_ms:.1f} ms，投影 {avg_projection_ms:.1f} ms。"
        )
    if avg_depth_ms > 90:
        score -= 8
        issues.append("深度估计稳态耗时偏高。")
        improvements.append("可降低 `runtime.process_width` 或 `model.depth_input_size`。")
    if avg_projection_ms > 10:
        score -= 6
        issues.append("CUDA 投影/拷贝耗时偏高。")
        improvements.append("可增大 `projection.stride`，或后续优化为显存内复用缓冲区。")

    ply_files = sorted((output_dir / "pointclouds").glob("*.ply"))
    if ply_files and ply_files[0].stat().st_size > 0:
        strengths.append(f"已生成 Open3D 点云文件：`{ply_files[0]}`。")
    else:
        score -= 8
        issues.append("未生成有效 PLY 点云。")
        improvements.append("检查点云数量和 `save_ply_interval`。")

    decision_path = output_dir / "decision" / "decision_report.txt"
    if decision_path.exists() and decision_path.stat().st_size > 0:
        strengths.append("已生成本地规则决策报告。")
    else:
        score -= 5
        issues.append("未生成决策报告。")
        improvements.append("检查 `outputs/decision/latest_scene_metrics.json` 是否存在。")

    target_frames = expected_frames or int(config["runtime"]["max_frames"])
    if frame_count < target_frames:
        score -= 6
        issues.append(f"本次只处理 {frame_count} 帧，少于本次目标 {target_frames} 帧。")
        improvements.append("检查输入视频长度，或降低目标帧数用于短验收。")

    score = max(0, min(100, score))
    level = "优秀" if score >= 88 else "良好" if score >= 75 else "需要改进"
    report = QualityReport(score=score, level=level, strengths=strengths, issues=issues, improvements=improvements)
    write_demo_report(report, output_dir)
    return report


def evaluate_benchmark_run(config: dict, rows: list[dict[str, Any]], benchmark_dir: str | Path) -> QualityReport:
    """评价 CPU/GPU 基准测试结果。"""

    benchmark_dir = resolve_path(benchmark_dir)
    strengths: list[str] = []
    issues: list[str] = []
    improvements: list[str] = []
    score = 100

    avg_loop = _average(rows, "cpu_loop_ms")
    avg_numpy = _average(rows, "cpu_numpy_ms")
    avg_gpu = _average(rows, "gpu_ms")
    avg_speedup = avg_loop / avg_gpu if avg_gpu > 0 else 0.0

    if avg_speedup >= 8.0:
        strengths.append(f"串行 CPU 到 CUDA 平均加速比约 {avg_speedup:.2f}x，适合作为并行加速展示。")
    elif avg_speedup >= 3.0:
        score -= 10
        issues.append(f"加速比一般：{avg_speedup:.2f}x。")
        improvements.append("可降低 `projection.stride`，增加像素并行规模，让 GPU 更充分。")
    else:
        score -= 25
        issues.append(f"加速比偏低：{avg_speedup:.2f}x。")
        improvements.append("检查是否退回 CPU，或使用更高分辨率/更小 stride 做基准。")

    strengths.append(f"平均耗时：CPU 串行 {avg_loop:.2f} ms，NumPy {avg_numpy:.2f} ms，CUDA {avg_gpu:.2f} ms。")
    if avg_numpy and avg_numpy < avg_gpu:
        strengths.append("NumPy 比 CUDA 更快属于正常工程现象，因为 CUDA 统计包含主机/显存拷贝。")

    csv_path = benchmark_dir / "results.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        score -= 15
        issues.append("未生成有效 benchmark CSV。")
        improvements.append("重新运行 `python -m src.benchmark --video data/input/demo.mp4 --max-frames 20`。")

    score = max(0, min(100, score))
    level = "优秀" if score >= 88 else "良好" if score >= 75 else "需要改进"
    report = QualityReport(score=score, level=level, strengths=strengths, issues=issues, improvements=improvements)
    write_benchmark_report(report, benchmark_dir)
    return report


def write_demo_report(report: QualityReport, output_dir: Path) -> None:
    """写出 run_demo 自评报告。"""

    report_path = output_dir / "self_evaluation.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# 运行效果自评\n\n")
        f.write("## 什么算较好的效果\n\n")
        f.write("- 深度热力图应与画面前后关系一致：近处目标颜色变化明显，远处背景连续。\n")
        f.write("- 语义图应能以 SAM 3 mask 覆盖主要人/车/障碍物，并在点云中呈现对应类别颜色。\n")
        f.write("- 点云俯视图应稳定、非空，前方空间结构能看出近远层次。\n")
        f.write("- 性能表应显示 CUDA 后端生效，并能给出串行 CPU 到 CUDA 的加速比。\n")
        f.write("- 决策报告应指出最近风险目标、是否处于中心通道以及建议动作。\n\n")
        _write_report_body(f, report)


def write_benchmark_report(report: QualityReport, benchmark_dir: Path) -> None:
    """写出 benchmark 自评报告。"""

    report_path = benchmark_dir / "self_evaluation.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# 基准测试自评\n\n")
        _write_report_body(f, report)


def _write_report_body(f, report: QualityReport) -> None:
    f.write(f"## 总评\n\n- 分数：{report.score}/100\n- 等级：{report.level}\n\n")
    f.write("## 做得好的地方\n\n")
    for item in report.strengths or ["暂无"]:
        f.write(f"- {item}\n")
    f.write("\n## 发现的问题\n\n")
    for item in report.issues or ["未发现明显问题。"]:
        f.write(f"- {item}\n")
    f.write("\n## 下一步自动改进建议\n\n")
    for item in report.improvements or ["当前指标稳定，可继续跑更长视频或接入真实 VLM。"]:
        f.write(f"- {item}\n")
