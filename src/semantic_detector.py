from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .utils import now_ms


@dataclass
class Detection:
    """单个语义检测框。"""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass
class SemanticResult:
    """语义检测输出。"""

    detections: list[Detection]
    label_mask: np.ndarray
    elapsed_ms: float
    backend: str


class SemanticDetector:
    """YOLO-World 开词表检测封装。

    YOLO-World 输出的是检测框而不是像素级分割。为了给点云着色，本项目采用
    一个简单但演示效果稳定的策略：检测框覆盖区域内的采样点继承该类别标签。
    这比 SAM2 全景分割轻很多，更适合 RTX 4060 上的课程演示。
    """

    def __init__(
        self,
        model_name: str,
        classes: list[str],
        conf: float = 0.18,
        imgsz: int = 640,
        device: str = "cuda",
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.enabled = enabled
        self.model = None

        if enabled:
            self._load_model()

    def _load_model(self) -> None:
        """加载 YOLO-World；若权重下载失败，后续自动返回空语义结果。"""

        try:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(self.model_name)
            self.model.set_classes(self.classes)
        except Exception as exc:
            print(f"YOLO-World 加载失败，将跳过语义检测：{exc}")
            self.model = None

    def predict(self, frame_bgr: np.ndarray) -> SemanticResult:
        """对一帧图像进行语义检测，并生成与图像同尺寸的标签掩码。"""

        start = now_ms()
        h, w = frame_bgr.shape[:2]
        label_mask = np.zeros((h, w), dtype=np.int32)
        detections: list[Detection] = []

        if self.model is None:
            return SemanticResult(detections, label_mask, now_ms() - start, backend="disabled")

        try:
            results = self.model.predict(
                frame_bgr,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            print(f"YOLO-World 推理失败，本帧跳过语义检测：{exc}")
            return SemanticResult(detections, label_mask, now_ms() - start, backend="failed")

        if not results:
            return SemanticResult(detections, label_mask, now_ms() - start, backend="yolo-world")

        boxes = results[0].boxes
        if boxes is None:
            return SemanticResult(detections, label_mask, now_ms() - start, backend="yolo-world")

        for xyxy, cls_id, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            class_index = int(cls_id)
            if class_index < 0 or class_index >= len(self.classes):
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # 标签 0 保留为“无语义”，类别从 1 开始，便于 CUDA kernel 用 0 判断背景点。
            label_id = class_index + 1
            label_mask[y1:y2, x1:x2] = label_id
            detections.append(
                Detection(
                    class_id=label_id,
                    class_name=self.classes[class_index],
                    confidence=float(conf),
                    xyxy=(x1, y1, x2, y2),
                )
            )

        return SemanticResult(detections, label_mask, now_ms() - start, backend="yolo-world")


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """把语义检测框画到图像上，便于和点云着色结果做对照。"""

    canvas = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = semantic_color(det.class_id).tolist()
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
    return canvas


def semantic_color(label_id: int) -> np.ndarray:
    """给语义标签分配稳定 RGB 颜色。"""

    palette = np.array(
        [
            [180, 180, 180],
            [255, 64, 64],
            [64, 180, 255],
            [255, 160, 64],
            [180, 96, 255],
            [64, 220, 120],
            [255, 220, 64],
            [255, 96, 180],
            [96, 255, 255],
        ],
        dtype=np.uint8,
    )
    return palette[label_id % len(palette)]


def build_color_lut(num_classes: int) -> np.ndarray:
    """构建 CUDA kernel 可直接索引的标签颜色表。"""

    lut = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    for label_id in range(num_classes + 1):
        lut[label_id] = semantic_color(label_id)
    return lut
