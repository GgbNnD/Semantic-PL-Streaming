from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import resolve_path
from .utils import now_ms


@dataclass
class DepthResult:
    """深度估计结果。

    Attributes:
        depth: 与输入帧同尺寸的浮点相对深度图。
        backend: 实际使用的后端，`official` 表示官方 Depth-Anything-V2，
            `fallback` 表示没有权重时使用的本地合成深度。
        elapsed_ms: 单帧深度估计耗时，单位毫秒。
    """

    depth: np.ndarray
    backend: str
    elapsed_ms: float


class FallbackDepthEstimator:
    """无权重兜底深度估计器。

    这个类不声称产生真实深度，只用于在模型下载失败时继续验证 CUDA 投影、
    可视化、性能统计和本地决策链路。它使用灰度、边缘和平滑构造一个稳定的
    相对深度场，最终报告中会标注为 fallback。
    """

    backend = "fallback"

    def predict(self, frame_bgr: np.ndarray) -> DepthResult:
        start = now_ms()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        # 近似假设：画面下方更接近摄像头，亮度和边缘提供局部起伏。
        h, w = gray.shape
        vertical_prior = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None]
        vertical_prior = np.repeat(vertical_prior, w, axis=1)
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        depth = 0.55 * vertical_prior + 0.35 * (gray / 255.0) + 0.10 * edges
        depth = cv2.GaussianBlur(depth, (7, 7), 0)
        return DepthResult(depth=depth.astype(np.float32), backend=self.backend, elapsed_ms=now_ms() - start)


class DepthAnythingV2Estimator:
    """官方 Depth-Anything-V2 推理封装。

    官方仓库的 `DepthAnythingV2.infer_image` 接收 OpenCV BGR 图像并返回与原图同
    尺寸的相对深度图。本封装只负责加载模型、设备选择和计时，避免业务代码直接
    依赖第三方仓库的路径结构。
    """

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    backend = "official"

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint_path: str | Path,
        encoder: str = "vits",
        input_size: int = 518,
        device: str = "cuda",
        allow_fallback: bool = True,
    ) -> None:
        self.repo_dir = resolve_path(repo_dir)
        self.checkpoint_path = resolve_path(checkpoint_path)
        self.encoder = encoder
        self.input_size = input_size
        self.device = self._select_device(device)
        self.fallback: FallbackDepthEstimator | None = FallbackDepthEstimator() if allow_fallback else None
        self.model = None

        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的 Depth-Anything-V2 encoder：{encoder}")
        self._load_model()

    def _select_device(self, requested: str) -> str:
        """根据配置和实际硬件选择推理设备。"""

        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        """加载官方仓库代码和权重；失败时按配置降级。"""

        if not self.repo_dir.exists() or not self.checkpoint_path.exists():
            if self.fallback is not None:
                print("Depth-Anything-V2 仓库或权重缺失，将使用 fallback 深度。请运行 python -m src.prepare_assets 下载正式模型。")
                return
            raise FileNotFoundError("Depth-Anything-V2 仓库或权重不存在")

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            self.model = DepthAnythingV2(**self.MODEL_CONFIGS[self.encoder])
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device).eval()
        except Exception as exc:
            if self.fallback is not None:
                print(f"Depth-Anything-V2 加载失败，将使用 fallback 深度：{exc}")
                self.model = None
                return
            raise

    def predict(self, frame_bgr: np.ndarray) -> DepthResult:
        """对单帧图像推理相对深度。"""

        if self.model is None:
            assert self.fallback is not None
            return self.fallback.predict(frame_bgr)

        start = now_ms()
        with torch.inference_mode():
            depth = self.model.infer_image(frame_bgr, self.input_size)
        depth = depth.astype(np.float32)
        return DepthResult(depth=depth, backend=self.backend, elapsed_ms=now_ms() - start)

