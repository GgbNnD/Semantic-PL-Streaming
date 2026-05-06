from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import ensure_project_dirs, load_config, resolve_path
from .decision import LocalRuleDecisionClient, compute_scene_metrics, save_decision_artifacts
from .depth_estimator import DepthAnythingV2Estimator
from .evaluation import evaluate_demo_run
from .filters import DepthFilter
from .live_viewer import Open3DLiveViewer
from .projector import PointCloudProjector
from .semantic_detector import SemanticDetector, draw_detections
from .utils import relative_depth_to_pseudo_z, resize_keep_aspect, safe_video_writer, write_csv
from .visualization import (
    compose_demo_panel,
    depth_to_heatmap,
    render_topdown,
    save_point_cloud_ply,
    save_point_cloud_preview,
    semantic_overlay,
)


def add_panel_title(image_bgr, title: str):
    """给演示面板加英文标题，避免 OpenCV 默认字体无法渲染中文。"""

    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return canvas


def clean_previous_outputs(output_dir: Path, keyframe_dir: Path, pointcloud_dir: Path, decision_dir: Path) -> None:
    """清理上一次 run_demo 生成的文件，避免短跑时混入旧关键帧。"""

    for path in [
        output_dir / "demo_side_by_side.mp4",
        output_dir / "performance.csv",
        output_dir / "run_summary.md",
        output_dir / "self_evaluation.md",
        decision_dir / "latest_scene_metrics.json",
        decision_dir / "decision_report.txt",
    ]:
        if path.exists():
            path.unlink()

    for folder, patterns in [
        (keyframe_dir, ["frame_*.jpg", "depth_*.jpg", "semantic_*.jpg", "pointcloud_*.jpg"]),
        (pointcloud_dir, ["cloud_*.ply"]),
    ]:
        for pattern in patterns:
            for path in folder.glob(pattern):
                path.unlink()


def run_demo(config: dict, video_path: Path, max_frames: int | None = None) -> None:
    """运行完整本地视觉管线。"""

    ensure_project_dirs(config)
    output_dir = resolve_path(config["paths"]["output_dir"])
    keyframe_dir = output_dir / "keyframes"
    pointcloud_dir = output_dir / "pointclouds"
    decision_dir = output_dir / "decision"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(output_dir, keyframe_dir, pointcloud_dir, decision_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}。请先运行 python -m src.prepare_assets 或放入自己的视频。")

    runtime = config["runtime"]
    model_cfg = config["model"]
    proj_cfg = config["projection"]
    filter_cfg = config["filtering"]
    decision_cfg = config["decision"]
    max_frames = max_frames or int(runtime["max_frames"])
    semantic_classes = model_cfg["sam3_classes"]

    depth_estimator = DepthAnythingV2Estimator(
        repo_dir=config["paths"]["depth_repo_dir"],
        checkpoint_path=config["paths"]["depth_checkpoint"],
        encoder=model_cfg["depth_encoder"],
        input_size=int(model_cfg["depth_input_size"]),
        device=runtime["device"],
        allow_fallback=bool(runtime["allow_fallback_depth"]),
    )
    semantic_detector = SemanticDetector(
        model_path=config["paths"].get("sam3_checkpoint", "third_party/sam3/sam3.pt"),
        classes=semantic_classes,
        conf=float(model_cfg["sam3_conf"]),
        imgsz=int(model_cfg["sam3_imgsz"]),
        device=runtime["device"],
        half=bool(model_cfg.get("sam3_half", True)),
        enabled=bool(runtime["use_sam3"]),
    )
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
    decision_client = LocalRuleDecisionClient(
        danger_z=float(decision_cfg["danger_z"]),
        warning_z=float(decision_cfg["warning_z"]),
    )
    viewer = Open3DLiveViewer(enabled=bool(config["visualization"]["live_open3d"]))

    rows: list[dict[str, object]] = []
    writer = None
    latest_metrics = {}
    latest_snapshot = None
    frame_index = 0
    output_video = output_dir / "demo_side_by_side.mp4"

    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, int(runtime["process_width"]))

        depth_result = depth_estimator.predict(frame)
        z_map = relative_depth_to_pseudo_z(
            depth_result.depth,
            min_z=float(proj_cfg["pseudo_min_z"]),
            max_z=float(proj_cfg["pseudo_max_z"]),
            invert=bool(proj_cfg["invert_depth"]),
        )
        filter_result = depth_filter.filter_auto(z_map, prefer_cuda=runtime["device"] == "cuda")
        z_map = filter_result.z_map

        semantic_result = semantic_detector.predict(frame)
        projection = projector.project_auto(
            z_map,
            frame,
            semantic_result.label_mask,
            prefer_cuda=runtime["device"] == "cuda",
        )
        viewer.update(projection.points_xyz, projection.colors_rgb)

        depth_vis = depth_to_heatmap(z_map)
        semantic_vis = semantic_overlay(draw_detections(frame, semantic_result.detections), semantic_result.label_mask)
        topdown_vis = render_topdown(projection.points_xyz, projection.colors_rgb)
        panel = compose_demo_panel(
            add_panel_title(frame, "Input video"),
            add_panel_title(depth_vis, f"Filtered depth ({filter_result.backend})"),
            add_panel_title(semantic_vis, f"Semantic ({semantic_result.backend})"),
            add_panel_title(topdown_vis, f"CUDA point cloud ({projection.backend})"),
        )

        if writer is None:
            writer = safe_video_writer(output_video, float(runtime["output_fps"]), (panel.shape[1], panel.shape[0]))
        writer.write(panel)

        if frame_index % int(runtime["keyframe_interval"]) == 0:
            latest_snapshot = keyframe_dir / f"frame_{frame_index:04d}.jpg"
            cv2.imwrite(str(latest_snapshot), panel)
            cv2.imwrite(str(keyframe_dir / f"depth_{frame_index:04d}.jpg"), depth_vis)
            cv2.imwrite(str(keyframe_dir / f"semantic_{frame_index:04d}.jpg"), semantic_vis)
            save_point_cloud_preview(keyframe_dir / f"pointcloud_{frame_index:04d}.jpg", projection.points_xyz, projection.colors_rgb)

        if frame_index % int(runtime["save_ply_interval"]) == 0:
            save_point_cloud_ply(pointcloud_dir / f"cloud_{frame_index:04d}.ply", projection.points_xyz, projection.colors_rgb)

        latest_metrics = compute_scene_metrics(
            z_map,
            semantic_result.detections,
            center_band_ratio=float(decision_cfg["center_band_ratio"]),
        )
        latest_metrics["frame_index"] = frame_index
        latest_metrics["depth_backend"] = depth_result.backend
        latest_metrics["filter_backend"] = filter_result.backend
        latest_metrics["projection_backend"] = projection.backend
        latest_metrics["point_count"] = int(projection.points_xyz.shape[0])

        rows.append(
            {
                "frame": frame_index,
                "depth_ms": round(depth_result.elapsed_ms, 3),
                "filter_ms": round(filter_result.elapsed_ms, 3),
                "semantic_ms": round(semantic_result.elapsed_ms, 3),
                "projection_ms": round(projection.elapsed_ms, 3),
                "point_count": int(projection.points_xyz.shape[0]),
                "depth_backend": depth_result.backend,
                "filter_backend": filter_result.backend,
                "semantic_backend": semantic_result.backend,
                "projection_backend": projection.backend,
            }
        )
        frame_index += 1
        if frame_index % 10 == 0:
            print(f"已处理 {frame_index} 帧")

    cap.release()
    viewer.close()
    if writer is not None:
        writer.release()

    performance_csv = output_dir / "performance.csv"
    write_csv(performance_csv, rows)
    decision_text = decision_client.analyze(latest_metrics, latest_snapshot)
    save_decision_artifacts(latest_metrics, decision_text, decision_dir)

    summary_path = output_dir / "run_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# 本地视觉管线运行摘要\n\n")
        f.write(f"- 输入视频：`{video_path}`\n")
        f.write(f"- 输出视频：`{output_video}`\n")
        f.write(f"- 性能 CSV：`{performance_csv}`\n")
        f.write(f"- 已处理帧数：{frame_index}\n")
        f.write(f"- 决策结果：\n\n{decision_text}\n")
        f.write("\n说明：当前深度是相对深度映射出的 Pseudo-LiDAR 伪距离，并非真实米制深度。\n")

    if bool(config["evaluation"]["enabled"]):
        report = evaluate_demo_run(config, rows, output_dir, frame_index, expected_frames=max_frames)
        print(f"自评报告：{output_dir / 'self_evaluation.md'}（{report.level}，{report.score}/100）")

    print(f"演示视频已生成：{output_video}")
    print(f"性能数据已生成：{performance_csv}")
    print(f"本地决策报告：{decision_dir / 'decision_report.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行本地视觉闭环演示")
    parser.add_argument("--config", default="configs/demo.yaml", help="配置文件路径")
    parser.add_argument("--video", default=None, help="输入视频路径，默认读取配置中的 data/input/demo.mp4")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = resolve_path(args.video or config["paths"]["input_video"])
    run_demo(config, video_path, args.max_frames)


if __name__ == "__main__":
    main()
