from __future__ import annotations

import unittest

import numpy as np

from src.filters import DepthFilter


class DepthFilterTest(unittest.TestCase):
    """验证深度统计滤波器的基本行为。"""

    def test_disabled_filter_returns_input(self) -> None:
        z_map = np.arange(25, dtype=np.float32).reshape(5, 5)
        depth_filter = DepthFilter(enabled=False)
        result = depth_filter.filter_auto(z_map, prefer_cuda=False)
        np.testing.assert_array_equal(result.z_map, z_map)
        self.assertEqual(result.backend, "disabled")

    def test_cuda_filter_preserves_shape(self) -> None:
        z_map = np.ones((12, 16), dtype=np.float32) * 5.0
        z_map[5, 6] = 20.0
        depth_filter = DepthFilter(enabled=True, radius=1, jump_threshold=1.0, min_neighbors=3)
        result = depth_filter.filter_auto(z_map, prefer_cuda=True)

        self.assertEqual(result.z_map.shape, z_map.shape)
        self.assertTrue(np.isfinite(result.z_map).all())
        self.assertIn(result.backend, {"cuda_filter", "cpu_bilateral"})


if __name__ == "__main__":
    unittest.main()

