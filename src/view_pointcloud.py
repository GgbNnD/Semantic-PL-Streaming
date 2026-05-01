from __future__ import annotations

import argparse
from pathlib import Path

from .config import resolve_path


def find_latest_ply(pointcloud_dir: Path) -> Path | None:
    """找到最新的 PLY 点云文件。"""

    files = sorted(pointcloud_dir.glob("*.ply"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Open3D 打开点云文件进行交互式查看")
    parser.add_argument("--ply", default=None, help="PLY 文件路径；不传则打开 outputs/pointclouds 下最新文件")
    args = parser.parse_args()

    ply_path = resolve_path(args.ply) if args.ply else find_latest_ply(resolve_path("outputs/pointclouds"))
    if ply_path is None or not ply_path.exists():
        raise FileNotFoundError("没有找到 PLY 点云文件，请先运行 python -m src.run_demo")

    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise RuntimeError(f"点云为空：{ply_path}")
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    print(f"正在打开点云：{ply_path}")
    print("鼠标左键旋转，滚轮缩放，右键平移。关闭窗口后命令结束。")
    o3d.visualization.draw_geometries([pcd, frame], window_name="Pseudo-LiDAR Point Cloud")


if __name__ == "__main__":
    main()
