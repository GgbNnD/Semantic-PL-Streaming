from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import now_ms

try:
    from numba import cuda
except Exception:  # pragma: no cover - 没有安装 numba 时只走 CPU 兜底
    cuda = None


@dataclass
class FilterResult:
    """深度/伪距离滤波结果。"""

    z_map: np.ndarray
    elapsed_ms: float
    backend: str


if cuda is not None:

    @cuda.jit
    def _edge_aware_mean_kernel(z_map, output, radius, jump_threshold, min_neighbors):
        """边缘保持均值滤波 CUDA kernel。

        每个线程负责一个像素。普通均值滤波会把前景障碍和背景距离混在一起，
        导致障碍物边缘被抹平；这里只统计与中心像素距离差不超过阈值的邻居，
        相当于一个简化的“统计学并行滤波器”。
        """

        x, y = cuda.grid(2)
        h = z_map.shape[0]
        w = z_map.shape[1]
        if x >= w or y >= h:
            return

        center = z_map[y, x]
        if center != center:
            output[y, x] = center
            return

        total = 0.0
        count = 0
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= h:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if xx < 0 or xx >= w:
                    continue
                value = z_map[yy, xx]
                if value == value and abs(value - center) <= jump_threshold:
                    total += value
                    count += 1

        if count >= min_neighbors:
            output[y, x] = total / count
        else:
            output[y, x] = center


class DepthFilter:
    """深度图统计滤波器。

    该滤波器位于深度估计和点云投影之间。它的目标不是追求最强去噪，而是在课程
    演示中清楚展示“每个像素独立做邻域统计”的并行计算价值，并减少深度突变造成
    的点云毛刺。
    """

    def __init__(
        self,
        enabled: bool = True,
        radius: int = 1,
        jump_threshold: float = 1.2,
        min_neighbors: int = 3,
    ) -> None:
        self.enabled = enabled
        self.radius = max(1, int(radius))
        self.jump_threshold = float(jump_threshold)
        self.min_neighbors = max(1, int(min_neighbors))

    def filter_cpu(self, z_map: np.ndarray) -> FilterResult:
        """CPU 兜底滤波。

        CPU 路径使用 OpenCV 双边滤波，语义上同样是“保边平滑”：距离差较大的区域
        不会被强行平均。它主要用于没有 CUDA 时保持管线可运行。
        """

        if not self.enabled:
            return FilterResult(z_map=z_map.astype(np.float32), elapsed_ms=0.0, backend="disabled")

        start = now_ms()
        z = z_map.astype(np.float32)
        diameter = self.radius * 2 + 1
        filtered = cv2.bilateralFilter(z, diameter, self.jump_threshold, diameter)
        return FilterResult(z_map=filtered.astype(np.float32), elapsed_ms=now_ms() - start, backend="cpu_bilateral")

    def filter_cuda(self, z_map: np.ndarray) -> FilterResult:
        """CUDA 边缘保持邻域均值滤波。"""

        if not self.enabled:
            return FilterResult(z_map=z_map.astype(np.float32), elapsed_ms=0.0, backend="disabled")
        if cuda is None or not cuda.is_available():
            return self.filter_cpu(z_map)

        start = now_ms()
        z = np.ascontiguousarray(z_map.astype(np.float32))
        d_z = cuda.to_device(z)
        d_out = cuda.device_array_like(d_z)
        block = (16, 16)
        grid = ((z.shape[1] + block[0] - 1) // block[0], (z.shape[0] + block[1] - 1) // block[1])
        _edge_aware_mean_kernel[grid, block](
            d_z,
            d_out,
            self.radius,
            self.jump_threshold,
            self.min_neighbors,
        )
        cuda.synchronize()
        filtered = d_out.copy_to_host()
        return FilterResult(z_map=filtered.astype(np.float32), elapsed_ms=now_ms() - start, backend="cuda_filter")

    def filter_auto(self, z_map: np.ndarray, prefer_cuda: bool = True) -> FilterResult:
        """根据硬件自动选择滤波后端。"""

        if prefer_cuda:
            return self.filter_cuda(z_map)
        return self.filter_cpu(z_map)

