from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import resolve_path
from .utils import image_grid, normalize_to_uint8


def depth_to_heatmap(depth: np.ndarray) -> np.ndarray:
    """把深度图转成 OpenCV BGR 热力图。

    这里使用 TURBO 色图：颜色变化明显，适合放到课程演示视频里观察近远关系。
    """

    depth_u8 = normalize_to_uint8(depth)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def semantic_overlay(frame_bgr: np.ndarray, label_mask: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """把语义标签半透明叠加到原图上。"""

    from .semantic_detector import semantic_color

    overlay = frame_bgr.copy()
    colored = np.zeros_like(frame_bgr)
    labels = np.unique(label_mask)
    for label_id in labels:
        if label_id <= 0:
            continue
        rgb = semantic_color(int(label_id))
        bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
        colored[label_mask == label_id] = bgr
    mask = label_mask > 0
    overlay[mask] = cv2.addWeighted(frame_bgr[mask], 1.0 - alpha, colored[mask], alpha, 0)
    return overlay


def render_topdown(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    """从点云生成俯视图。

    这不是严格的 3D 渲染，而是把 X-Z 平面投到一张 2D 图上。它计算快、稳定，
    适合作为每帧演示视频的点云视图；真正的 3D 点云会另存为 PLY 和关键帧散点图。
    """

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if points_xyz.size == 0:
        cv2.putText(canvas, "empty point cloud", (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return canvas

    x = points_xyz[:, 0]
    z = points_xyz[:, 2]
    valid = np.isfinite(x) & np.isfinite(z)
    x = x[valid]
    z = z[valid]
    colors = colors_rgb[valid]
    if x.size == 0:
        return canvas

    z_min = float(np.percentile(z, 1))
    z_max = float(np.percentile(z, 99))
    if z_max <= z_min:
        z_max = z_min + 1.0
    x_abs = max(float(np.percentile(np.abs(x), 98)), 1.0)

    px = ((x + x_abs) / (2.0 * x_abs) * (width - 1)).astype(np.int32)
    # 近处放在图像底部，远处放在顶部，更符合俯视雷达图的直觉。
    py = ((1.0 - (z - z_min) / (z_max - z_min)) * (height - 1)).astype(np.int32)
    keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[keep]
    py = py[keep]
    bgr = colors[keep][:, ::-1]
    canvas[py, px] = bgr

    # 画出摄像头位置和前进方向，帮助观众理解坐标系。
    cv2.circle(canvas, (width // 2, height - 18), 7, (255, 255, 255), -1)
    cv2.line(canvas, (width // 2, height - 25), (width // 2, height - 70), (255, 255, 255), 2)
    cv2.putText(canvas, "Pseudo-LiDAR top view", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230), 2)
    return canvas


def compose_demo_panel(
    frame_bgr: np.ndarray,
    depth_heatmap_bgr: np.ndarray,
    semantic_bgr: np.ndarray,
    point_view_bgr: np.ndarray,
) -> np.ndarray:
    """生成 2x2 演示面板。"""

    return image_grid([frame_bgr, depth_heatmap_bgr, semantic_bgr, point_view_bgr], cols=2, cell_width=640)


def save_point_cloud_ply(path: str | Path, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> bool:
    """用 Open3D 保存 PLY 点云文件。

    Open3D 在无显示器服务器上离屏渲染可能不稳定，但写 PLY 文件通常可靠。这样既
    满足 3D 处理工具链要求，也方便用户之后用 Open3D/CloudCompare 手动打开检查。
    """

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if points_xyz.size == 0:
        return False
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((colors_rgb.astype(np.float64) / 255.0).clip(0.0, 1.0))
        return bool(o3d.io.write_point_cloud(str(path), pcd))
    except Exception as exc:
        print(f"Open3D 保存 PLY 失败：{exc}")
        return False


def save_point_cloud_preview(
    path: str | Path,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    max_points: int = 20000,
) -> None:
    """保存 3D 散点图预览，用于关键帧材料。"""

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if points_xyz.size == 0:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(path), blank)
        return

    if points_xyz.shape[0] > max_points:
        indices = np.linspace(0, points_xyz.shape[0] - 1, max_points).astype(np.int64)
        points = points_xyz[indices]
        colors = colors_rgb[indices] / 255.0
    else:
        points = points_xyz
        colors = colors_rgb / 255.0

    fig = plt.figure(figsize=(7.2, 5.0), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=colors, s=0.6, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    ax.set_title("Pseudo-LiDAR point cloud")
    ax.view_init(elev=18, azim=-72)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

