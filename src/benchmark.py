from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import ensure_project_dirs, load_config, resolve_path
from .evaluation import evaluate_benchmark_run
from .filters import DepthFilter
from .projector import PointCloudProjector
from .utils import relative_depth_to_pseudo_z, resize_keep_aspect, write_csv


def frame_to_fast_depth(frame_bgr: np.ndarray, min_z: float, max_z: float, invert: bool) -> np.ndarray:
    """为几何层基准测试快速构造伪深度。

    benchmark 的目标是隔离 2D→3D 投影速度，所以不加载深度网络；用灰度图构造
    稳定输入即可比较 CPU baseline 和 CUDA kernel 的耗时。
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return relative_depth_to_pseudo_z(gray, min_z=min_z, max_z=max_z, invert=invert)


def synthetic_frame(width: int = 960, height: int = 540) -> np.ndarray:
    """没有视频时生成一张合成测试图，保证 benchmark 可独立运行。"""

    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = 255 - x
    return frame


def run_benchmark(config: dict, video_path: Path, max_frames: int) -> None:
    ensure_project_dirs(config)
    proj_cfg = config["projection"]
    filter_cfg = config["filtering"]
    runtime = config["runtime"]
    model_cfg = config["model"]
    semantic_classes = model_cfg["sam3_classes"]

    projector = PointCloudProjector(
        stride=int(proj_cfg["stride"]),
        fov_degrees=float(proj_cfg["fov_degrees"]),
        min_z=float(proj_cfg["filter_min_z"]),
        max_z=float(proj_cfg["filter_max_z"]),
        num_classes=len(semantic_classes),
    )
    depth_filter = DepthFilter(
        enabled=bool(filter_cfg["enabled"]),
        radius=int(filter_cfg["radius"]),
        jump_threshold=float(filter_cfg["depth_jump_threshold"]),
        min_neighbors=int(filter_cfg["min_neighbors"]),
    )

    cap = cv2.VideoCapture(str(video_path))
    use_video = cap.isOpened()
    rows: list[dict[str, object]] = []

    # 先做一次 GPU warm-up，避免 JIT 编译时间污染正式统计。
    warm = synthetic_frame()
    warm_z = frame_to_fast_depth(warm, float(proj_cfg["pseudo_min_z"]), float(proj_cfg["pseudo_max_z"]), bool(proj_cfg["invert_depth"]))
    depth_filter.filter_cuda(warm_z)
    projector.project_cuda(warm_z, warm)

    for frame_index in range(max_frames):
        if use_video:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_keep_aspect(frame, int(runtime["process_width"]))
        else:
            frame = synthetic_frame()

        z_map = frame_to_fast_depth(
            frame,
            min_z=float(proj_cfg["pseudo_min_z"]),
            max_z=float(proj_cfg["pseudo_max_z"]),
            invert=bool(proj_cfg["invert_depth"]),
        )
        filtered = depth_filter.filter_auto(z_map, prefer_cuda=runtime["device"] == "cuda")
        z_map = filtered.z_map
        cpu_loop = projector.project_cpu_loop(z_map, frame)
        cpu_numpy = projector.project_cpu(z_map, frame)
        gpu = projector.project_cuda(z_map, frame)
        speedup = cpu_loop.elapsed_ms / gpu.elapsed_ms if gpu.elapsed_ms > 0 else 0.0
        rows.append(
            {
                "frame": frame_index,
                "cpu_loop_ms": round(cpu_loop.elapsed_ms, 3),
                "cpu_numpy_ms": round(cpu_numpy.elapsed_ms, 3),
                "filter_ms": round(filtered.elapsed_ms, 3),
                "filter_backend": filtered.backend,
                "gpu_ms": round(gpu.elapsed_ms, 3),
                "speedup": round(speedup, 3),
                "points": int(gpu.points_xyz.shape[0]),
                "gpu_backend": gpu.backend,
            }
        )

    if use_video:
        cap.release()

    benchmark_dir = resolve_path(config["paths"]["benchmark_dir"])
    csv_path = benchmark_dir / "results.csv"
    write_csv(csv_path, rows)

    if rows:
        avg_cpu = float(np.mean([row["cpu_loop_ms"] for row in rows]))
        avg_numpy = float(np.mean([row["cpu_numpy_ms"] for row in rows]))
        avg_filter = float(np.mean([row["filter_ms"] for row in rows]))
        avg_gpu = float(np.mean([row["gpu_ms"] for row in rows]))
        avg_speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0.0
        print(f"CPU 串行平均耗时：{avg_cpu:.3f} ms")
        print(f"CPU NumPy 平均耗时：{avg_numpy:.3f} ms")
        print(f"GPU 滤波平均耗时：{avg_filter:.3f} ms")
        print(f"GPU CUDA 平均耗时：{avg_gpu:.3f} ms")
        print(f"串行 CPU → CUDA 平均加速比：{avg_speedup:.2f}x")
    if bool(config["evaluation"]["enabled"]):
        report = evaluate_benchmark_run(config, rows, benchmark_dir)
        print(f"基准自评报告：{benchmark_dir / 'self_evaluation.md'}（{report.level}，{report.score}/100）")
    print(f"基准测试结果已写入：{csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU/GPU 点云反投影性能对比")
    parser.add_argument("--config", default="configs/demo.yaml", help="配置文件路径")
    parser.add_argument("--video", default=None, help="输入视频路径；不存在时自动使用合成帧")
    parser.add_argument("--max-frames", type=int, default=60, help="测试帧数")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = resolve_path(args.video or config["paths"]["input_video"])
    run_benchmark(config, video_path, args.max_frames)


if __name__ == "__main__":
    main()
