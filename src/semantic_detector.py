from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from .config import resolve_path
from .utils import now_ms


SAM3_MIN_ULTRALYTICS = (8, 3, 237)


@dataclass
class Detection:
    """单个语义实例，`xyxy` 是该实例 mask 的外接框。"""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass
class SemanticResult:
    """语义分割输出。"""

    detections: list[Detection]
    label_mask: np.ndarray
    elapsed_ms: float
    backend: str


class SemanticDetector:
    """Ultralytics SAM 3 文本概念分割封装。

    SAM 3 直接输出每个文本概念对应的实例 mask。本项目把这些 mask 合成为
    与输入图像同尺寸的 `label_mask`，后续 CUDA 反投影可以逐像素给点云着色。
    """

    def __init__(
        self,
        model_path: str | Path,
        classes: list[str],
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "cuda",
        half: bool = True,
        enabled: bool = True,
    ) -> None:
        self.model_path = model_path
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        self.device = "0" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.half = bool(half and self.device != "cpu")
        self.enabled = enabled
        self.model = None

        if enabled:
            self._load_model()

    def _load_model(self) -> None:
        """加载 SAM 3；若环境或权重不可用，后续自动返回空语义结果。"""

        checkpoint_path = resolve_path(self.model_path)
        if not checkpoint_path.exists():
            print(f"SAM 3 权重不存在，将跳过语义分割：{checkpoint_path}")
            self.model = None
            return

        try:
            import ultralytics
            from ultralytics.models.sam import SAM3SemanticPredictor

            if not _version_at_least(ultralytics.__version__, SAM3_MIN_ULTRALYTICS):
                required = ".".join(str(part) for part in SAM3_MIN_ULTRALYTICS)
                raise RuntimeError(f"Ultralytics {required}+ 才支持 SAM 3，当前版本为 {ultralytics.__version__}")

            overrides = {
                "conf": self.conf,
                "task": "segment",
                "mode": "predict",
                "model": str(checkpoint_path),
                "imgsz": self.imgsz,
                "device": self.device,
                "half": self.half,
                "save": False,
                "verbose": False,
            }
            self.model = SAM3SemanticPredictor(overrides=overrides)
            self.model.setup_model(verbose=False)
        except Exception as exc:
            print(f"SAM 3 加载失败，将跳过语义分割：{exc}")
            self.model = None

    def predict(self, frame_bgr: np.ndarray) -> SemanticResult:
        """对一帧图像进行 SAM 3 文本概念分割，并生成标签掩码。"""

        start = now_ms()
        h, w = frame_bgr.shape[:2]
        label_mask = np.zeros((h, w), dtype=np.int32)
        detections: list[Detection] = []

        if self.model is None:
            return SemanticResult(detections, label_mask, now_ms() - start, backend="disabled")

        try:
            self.model.set_image(frame_bgr)
            results = self.model(text=self.classes)
        except Exception as exc:
            print(f"SAM 3 推理失败，本帧跳过语义分割：{exc}")
            return SemanticResult(detections, label_mask, now_ms() - start, backend="failed")
        finally:
            try:
                self.model.reset_image()
            except Exception:
                pass

        if not results:
            return SemanticResult(detections, label_mask, now_ms() - start, backend="sam3")

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return SemanticResult(detections, label_mask, now_ms() - start, backend="sam3")

        xyxys = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxys), dtype=np.float32)
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxys), dtype=np.float32)
        masks = _extract_masks(result, (h, w))
        count = min(len(xyxys), len(class_ids), len(confs))

        for index in np.argsort(confs[:count]):
            xyxy = xyxys[index]
            class_index = int(class_ids[index])
            if class_index < 0 or class_index >= len(self.classes):
                continue
            x1, y1, x2, y2 = _clip_xyxy(xyxy, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            label_id = class_index + 1
            if masks is not None and index < masks.shape[0]:
                instance_mask = masks[index]
                if not instance_mask.any():
                    continue
                # 标签 0 保留为背景，类别从 1 开始，便于 CUDA kernel 快速判断语义点。
                label_mask[instance_mask] = label_id
            else:
                label_mask[y1:y2, x1:x2] = label_id
            detections.append(
                Detection(
                    class_id=label_id,
                    class_name=self.classes[class_index],
                    confidence=float(confs[index]),
                    xyxy=(x1, y1, x2, y2),
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return SemanticResult(detections, label_mask, now_ms() - start, backend="sam3")


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """把语义实例外接框画到图像上，便于和 mask 着色结果做对照。"""

    canvas = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        x2 = max(x1, min(canvas.shape[1] - 1, x2 - 1))
        y2 = max(y1, min(canvas.shape[0] - 1, y2 - 1))
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


def _version_at_least(version: str, minimum: tuple[int, int, int]) -> bool:
    parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
    parts += [0] * (3 - len(parts))
    return tuple(parts[:3]) >= minimum


def _clip_xyxy(xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def _extract_masks(result, target_shape: tuple[int, int]) -> np.ndarray | None:
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None:
        return None

    masks = masks_obj.data.cpu().numpy().astype(bool)
    target_h, target_w = target_shape
    if masks.ndim != 3:
        return None
    if masks.shape[1:] == target_shape:
        return masks

    resized = []
    for mask in masks:
        resized.append(cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0)
    return np.stack(resized, axis=0) if resized else None
