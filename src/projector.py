from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .semantic_detector import build_color_lut
from .utils import now_ms

try:
    from numba import cuda
except Exception:  # pragma: no cover - 仅在没有安装 numba 时触发
    cuda = None


@dataclass
class CameraIntrinsics:
    """针孔相机内参。

    fx/fy 是焦距的像素表示，cx/cy 是主点。没有真实标定文件时，本项目用水平
    FOV 近似反推 fx，并令 fy=fx；这足够展示 2D 像素到 3D 射线的并行反投影。
    """

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ProjectionResult:
    """点云反投影结果。"""

    points_xyz: np.ndarray
    colors_rgb: np.ndarray
    labels: np.ndarray
    elapsed_ms: float
    backend: str


def build_intrinsics(width: int, height: int, fov_degrees: float) -> CameraIntrinsics:
    """根据图像尺寸和水平视场角构造近似相机内参。"""

    fov_radians = math.radians(fov_degrees)
    fx = (0.5 * width) / math.tan(0.5 * fov_radians)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


if cuda is not None:

    @cuda.jit
    def _back_project_kernel(
        z_map,
        frame_bgr,
        label_mask,
        color_lut,
        points,
        colors,
        labels_out,
        valid,
        stride,
        fx,
        fy,
        cx,
        cy,
        min_z,
        max_z,
    ):
        """CUDA 反投影核函数。

        每个线程负责一个采样像素 `(u, v)`。线程之间没有共享状态，因此这是典型
        的数据并行任务：读取深度 Z 和颜色/标签，按针孔相机公式写出一个 3D 点。
        """

        ix, iy = cuda.grid(2)
        sampled_h = points.shape[0]
        sampled_w = points.shape[1]
        if ix >= sampled_w or iy >= sampled_h:
            return

        v = iy * stride
        u = ix * stride
        if v >= z_map.shape[0] or u >= z_map.shape[1]:
            return

        z = z_map[v, u]
        if z < min_z or z > max_z:
            return

        # 针孔模型反投影公式：
        # X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth。
        points[iy, ix, 0] = (u - cx) * z / fx
        points[iy, ix, 1] = (v - cy) * z / fy
        points[iy, ix, 2] = z

        label_id = label_mask[v, u]
        labels_out[iy, ix] = label_id
        if label_id > 0 and label_id < color_lut.shape[0]:
            colors[iy, ix, 0] = color_lut[label_id, 0]
            colors[iy, ix, 1] = color_lut[label_id, 1]
            colors[iy, ix, 2] = color_lut[label_id, 2]
        else:
            # OpenCV 图像是 BGR，点云可视化使用 RGB，因此这里顺手翻转通道。
            colors[iy, ix, 0] = frame_bgr[v, u, 2]
            colors[iy, ix, 1] = frame_bgr[v, u, 1]
            colors[iy, ix, 2] = frame_bgr[v, u, 0]
        valid[iy, ix] = 1


class PointCloudProjector:
    """2D 深度图到 3D 点云的 CPU/GPU 反投影器。"""

    def __init__(
        self,
        stride: int,
        fov_degrees: float,
        min_z: float,
        max_z: float,
        num_classes: int,
    ) -> None:
        self.stride = max(1, int(stride))
        self.fov_degrees = float(fov_degrees)
        self.min_z = float(min_z)
        self.max_z = float(max_z)
        self.color_lut = build_color_lut(num_classes)

    def project_cpu(
        self,
        z_map: np.ndarray,
        frame_bgr: np.ndarray,
        label_mask: np.ndarray | None = None,
    ) -> ProjectionResult:
        """NumPy CPU baseline，用于和 CUDA kernel 做速度对比。"""

        start = now_ms()
        h, w = z_map.shape
        intr = build_intrinsics(w, h, self.fov_degrees)
        if label_mask is None:
            label_mask = np.zeros((h, w), dtype=np.int32)

        ys = np.arange(0, h, self.stride, dtype=np.float32)
        xs = np.arange(0, w, self.stride, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(xs, ys)
        sampled_z = z_map[:: self.stride, :: self.stride].astype(np.float32)
        sampled_labels = label_mask[:: self.stride, :: self.stride].astype(np.int32)
        sampled_bgr = frame_bgr[:: self.stride, :: self.stride]

        valid = (sampled_z >= self.min_z) & (sampled_z <= self.max_z) & np.isfinite(sampled_z)
        x = (u_grid - intr.cx) * sampled_z / intr.fx
        y = (v_grid - intr.cy) * sampled_z / intr.fy
        points = np.stack([x, y, sampled_z], axis=-1)[valid]

        colors_rgb = sampled_bgr[..., ::-1].copy()
        semantic_pixels = sampled_labels > 0
        if semantic_pixels.any():
            clipped = np.clip(sampled_labels[semantic_pixels], 0, self.color_lut.shape[0] - 1)
            colors_rgb[semantic_pixels] = self.color_lut[clipped]
        colors = colors_rgb[valid].astype(np.uint8)
        labels = sampled_labels[valid]

        return ProjectionResult(points, colors, labels, now_ms() - start, backend="cpu")

    def project_cpu_loop(
        self,
        z_map: np.ndarray,
        frame_bgr: np.ndarray,
        label_mask: np.ndarray | None = None,
    ) -> ProjectionResult:
        """纯 Python 串行 baseline。

        NumPy 向量化已经会调用底层 C 实现，不适合用来说明“每个像素独立反投影”
        这种数据并行任务的 CUDA 加速价值。因此 benchmark 使用这个逐点循环版本
        作为开启 CUDA 前的朴素实现；`project_cpu` 仍保留为工程上的高效 CPU 参考。
        """

        start = now_ms()
        h, w = z_map.shape
        intr = build_intrinsics(w, h, self.fov_degrees)
        if label_mask is None:
            label_mask = np.zeros((h, w), dtype=np.int32)

        points: list[tuple[float, float, float]] = []
        colors: list[tuple[int, int, int]] = []
        labels: list[int] = []

        for v in range(0, h, self.stride):
            for u in range(0, w, self.stride):
                z = float(z_map[v, u])
                if not math.isfinite(z) or z < self.min_z or z > self.max_z:
                    continue
                x = (u - intr.cx) * z / intr.fx
                y = (v - intr.cy) * z / intr.fy
                label_id = int(label_mask[v, u])
                if label_id > 0 and label_id < self.color_lut.shape[0]:
                    rgb = self.color_lut[label_id]
                    color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                else:
                    b, g, r = frame_bgr[v, u]
                    color = (int(r), int(g), int(b))
                points.append((x, y, z))
                colors.append(color)
                labels.append(label_id)

        return ProjectionResult(
            points_xyz=np.asarray(points, dtype=np.float32),
            colors_rgb=np.asarray(colors, dtype=np.uint8),
            labels=np.asarray(labels, dtype=np.int32),
            elapsed_ms=now_ms() - start,
            backend="cpu_loop",
        )

    def project_cuda(
        self,
        z_map: np.ndarray,
        frame_bgr: np.ndarray,
        label_mask: np.ndarray | None = None,
    ) -> ProjectionResult:
        """Numba CUDA 反投影。

        这里把深度图、原图和语义标签传到显存，再由每个线程独立完成一个采样点的
        反投影。计时包含主机/显存拷贝，更贴近日常演示中的端到端几何层耗时。
        """

        if cuda is None or not cuda.is_available():
            return self.project_cpu(z_map, frame_bgr, label_mask)

        start = now_ms()
        h, w = z_map.shape
        intr = build_intrinsics(w, h, self.fov_degrees)
        if label_mask is None:
            label_mask = np.zeros((h, w), dtype=np.int32)

        z_map = np.ascontiguousarray(z_map.astype(np.float32))
        frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
        label_mask = np.ascontiguousarray(label_mask.astype(np.int32))
        sampled_h = (h + self.stride - 1) // self.stride
        sampled_w = (w + self.stride - 1) // self.stride

        d_z = cuda.to_device(z_map)
        d_frame = cuda.to_device(frame_bgr)
        d_labels = cuda.to_device(label_mask)
        d_lut = cuda.to_device(self.color_lut)
        d_points = cuda.device_array((sampled_h, sampled_w, 3), dtype=np.float32)
        d_colors = cuda.device_array((sampled_h, sampled_w, 3), dtype=np.uint8)
        d_labels_out = cuda.device_array((sampled_h, sampled_w), dtype=np.int32)
        d_valid = cuda.device_array((sampled_h, sampled_w), dtype=np.uint8)
        d_valid[:] = 0

        block = (16, 16)
        grid = ((sampled_w + block[0] - 1) // block[0], (sampled_h + block[1] - 1) // block[1])
        _back_project_kernel[grid, block](
            d_z,
            d_frame,
            d_labels,
            d_lut,
            d_points,
            d_colors,
            d_labels_out,
            d_valid,
            self.stride,
            intr.fx,
            intr.fy,
            intr.cx,
            intr.cy,
            self.min_z,
            self.max_z,
        )
        cuda.synchronize()

        points_host = d_points.copy_to_host()
        colors_host = d_colors.copy_to_host()
        labels_host = d_labels_out.copy_to_host()
        valid_host = d_valid.copy_to_host().astype(bool)
        elapsed = now_ms() - start

        return ProjectionResult(
            points_xyz=points_host[valid_host],
            colors_rgb=colors_host[valid_host],
            labels=labels_host[valid_host],
            elapsed_ms=elapsed,
            backend="cuda",
        )

    def project_auto(
        self,
        z_map: np.ndarray,
        frame_bgr: np.ndarray,
        label_mask: np.ndarray | None = None,
        prefer_cuda: bool = True,
    ) -> ProjectionResult:
        """根据硬件自动选择 CUDA 或 CPU。"""

        if prefer_cuda:
            return self.project_cuda(z_map, frame_bgr, label_mask)
        return self.project_cpu(z_map, frame_bgr, label_mask)
