from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import ensure_project_dirs, load_config, resolve_path
from .utils import download_file, run_command


DEPTH_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_VITS_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"
OPENCV_VTEST_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"


def clone_depth_repo(repo_dir: Path) -> None:
    """克隆官方 Depth-Anything-V2 仓库。"""

    if (repo_dir / "depth_anything_v2").exists():
        print(f"Depth-Anything-V2 仓库已存在：{repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"克隆 Depth-Anything-V2 到 {repo_dir}")
    run_command(["git", "clone", "--depth", "1", DEPTH_REPO_URL, str(repo_dir)])


def prepare_depth_checkpoint(checkpoint_path: Path, repo_dir: Path) -> None:
    """下载 vits 权重，并同步到官方仓库 checkpoints 目录。"""

    if checkpoint_path.exists():
        print(f"Depth-Anything-V2 vits 权重已存在：{checkpoint_path}")
    else:
        download_file(DEPTH_VITS_URL, checkpoint_path)

    official_checkpoint = repo_dir / "checkpoints" / checkpoint_path.name
    official_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not official_checkpoint.exists():
        official_checkpoint.write_bytes(checkpoint_path.read_bytes())
        print(f"已复制权重到官方仓库目录：{official_checkpoint}")


def convert_avi_to_mp4(source_avi: Path, target_mp4: Path) -> None:
    """把 OpenCV 官方 AVI 样例转成 MP4，统一后续命令的默认输入路径。"""

    if target_mp4.exists():
        print(f"演示视频已存在：{target_mp4}")
        return

    cap = cv2.VideoCapture(str(source_avi))
    if not cap.isOpened():
        print(f"无法打开备用视频：{source_avi}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"无法创建 MP4 输出：{target_mp4}")
        cap.release()
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"已生成备用演示视频：{target_mp4}")


def prepare_demo_video(input_video: Path) -> None:
    """准备默认演示视频。

    Pexels/Pixabay 页面下载通常需要网页交互或动态链接，不适合写成稳定脚本。
    因此本命令使用 OpenCV 官方可直链样例作为自动兜底；你之后可直接把自己的
    道路交通视频覆盖到 `data/input/demo.mp4`。
    """

    if input_video.exists():
        print(f"演示视频已存在：{input_video}")
        return
    avi_path = input_video.with_name("vtest.avi")
    if not avi_path.exists():
        download_file(OPENCV_VTEST_URL, avi_path)
    convert_avi_to_mp4(avi_path, input_video)


def warmup_yolo(model_name: str, classes: list[str]) -> None:
    """提前触发 YOLO-World 权重下载，减少第一次 run_demo 的等待时间。"""

    try:
        from ultralytics import YOLOWorld

        model = YOLOWorld(model_name)
        model.set_classes(classes)
        print(f"YOLO-World 权重准备完成：{model_name}")
    except Exception as exc:
        print(f"YOLO-World 权重准备失败，run_demo 会自动降级跳过语义：{exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备模型权重和演示素材")
    parser.add_argument("--config", default="configs/demo.yaml", help="配置文件路径")
    parser.add_argument("--skip-yolo", action="store_true", help="跳过 YOLO-World 权重预热")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    repo_dir = resolve_path(config["paths"]["depth_repo_dir"])
    checkpoint_path = resolve_path(config["paths"]["depth_checkpoint"])
    input_video = resolve_path(config["paths"]["input_video"])

    clone_depth_repo(repo_dir)
    prepare_depth_checkpoint(checkpoint_path, repo_dir)
    prepare_demo_video(input_video)

    if not args.skip_yolo:
        warmup_yolo(config["model"]["yolo_model"], config["model"]["yolo_classes"])

    print("资产准备完成。")


if __name__ == "__main__":
    main()

