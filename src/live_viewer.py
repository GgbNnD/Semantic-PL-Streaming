from __future__ import annotations

import numpy as np


class Open3DLiveViewer:
    """可选 Open3D 实时点云窗口。

    默认配置不会打开窗口，因为很多开发环境是无显示器或远程终端。若在本地桌面
    演示，可把 `visualization.live_open3d` 设为 true，程序会在每帧更新交互式
    点云视图，支持鼠标旋转、缩放和漫游。
    """

    def __init__(self, enabled: bool = False, window_name: str = "Pseudo-LiDAR Live View") -> None:
        self.enabled = enabled
        self.window_name = window_name
        self.available = False
        self._vis = None
        self._pcd = None
        self._created_geometry = False
        if enabled:
            self._init_viewer()

    def _init_viewer(self) -> None:
        try:
            import open3d as o3d

            self._pcd = o3d.geometry.PointCloud()
            self._vis = o3d.visualization.Visualizer()
            self.available = bool(self._vis.create_window(self.window_name, width=960, height=720, visible=True))
            if not self.available:
                print("Open3D 实时窗口创建失败，将继续保存离线点云。")
        except Exception as exc:
            print(f"Open3D 实时窗口不可用，将继续保存离线点云：{exc}")
            self.available = False

    def update(self, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
        """用当前帧点云刷新 Open3D 窗口。"""

        if not self.available or self._vis is None or self._pcd is None or points_xyz.size == 0:
            return
        try:
            import open3d as o3d

            self._pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
            self._pcd.colors = o3d.utility.Vector3dVector((colors_rgb.astype(np.float64) / 255.0).clip(0.0, 1.0))
            if not self._created_geometry:
                self._vis.add_geometry(self._pcd)
                self._created_geometry = True
            else:
                self._vis.update_geometry(self._pcd)
            self._vis.poll_events()
            self._vis.update_renderer()
        except Exception as exc:
            print(f"Open3D 实时刷新失败，关闭实时窗口：{exc}")
            self.close()

    def close(self) -> None:
        """关闭窗口并释放资源。"""

        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
        self.available = False

