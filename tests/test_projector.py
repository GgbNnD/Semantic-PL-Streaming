from __future__ import annotations

import unittest

import numpy as np

from src.projector import PointCloudProjector


class ProjectorTest(unittest.TestCase):
    """验证 CPU baseline 和 CUDA kernel 的几何结果一致。"""

    def test_cpu_projection_shape_and_labels(self) -> None:
        z_map = np.ones((6, 8), dtype=np.float32) * 5.0
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        frame[..., 2] = 255
        labels = np.zeros((6, 8), dtype=np.int32)
        labels[2:5, 3:7] = 1

        projector = PointCloudProjector(stride=2, fov_degrees=70.0, min_z=0.3, max_z=20.0, num_classes=3)
        result = projector.project_cpu(z_map, frame, labels)

        self.assertGreater(result.points_xyz.shape[0], 0)
        self.assertEqual(result.points_xyz.shape[1], 3)
        self.assertEqual(result.colors_rgb.shape[1], 3)
        self.assertTrue((result.labels >= 0).all())

    def test_cuda_matches_cpu_when_available(self) -> None:
        z_map = np.linspace(1.0, 10.0, 80, dtype=np.float32).reshape(8, 10)
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        frame[..., 0] = 10
        frame[..., 1] = 80
        frame[..., 2] = 200
        labels = np.zeros((8, 10), dtype=np.int32)
        labels[:, 5:] = 2

        projector = PointCloudProjector(stride=2, fov_degrees=70.0, min_z=0.3, max_z=20.0, num_classes=4)
        cpu = projector.project_cpu(z_map, frame, labels)
        gpu = projector.project_cuda(z_map, frame, labels)

        # 没有 CUDA 时 project_cuda 会自动降级到 CPU；有 CUDA 时比较同一采样顺序下的坐标。
        self.assertEqual(cpu.points_xyz.shape, gpu.points_xyz.shape)
        np.testing.assert_allclose(cpu.points_xyz, gpu.points_xyz, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(cpu.labels, gpu.labels)


if __name__ == "__main__":
    unittest.main()

