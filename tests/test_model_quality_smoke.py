from __future__ import annotations

import os
import unittest

import cv2
import numpy as np

from src.config import load_config, resolve_path
from src.depth_estimator import DepthAnythingV2Estimator
from src.semantic_detector import SemanticDetector


RUN_HEAVY_MODEL_TESTS = os.getenv("RUN_HEAVY_MODEL_TESTS") == "1"


def _sample_frame(width: int = 320) -> np.ndarray:
    config = load_config("configs/demo.yaml")
    video_path = resolve_path(config["paths"]["input_video"])
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if ok and frame is not None:
        scale = width / frame.shape[1]
        return cv2.resize(frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)

    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, width // 2, dtype=np.uint8)[:, None]
    frame = np.zeros((width // 2, width, 3), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = 255 - x
    return frame


@unittest.skipUnless(RUN_HEAVY_MODEL_TESTS, "设置 RUN_HEAVY_MODEL_TESTS=1 才加载 vitl/SAM 3 大模型")
class HeavyModelQualitySmokeTest(unittest.TestCase):
    """真实模型的轻量 smoke test。

    这类测试会加载 1.3GB 的 Depth-Anything-V2 vitl 和 3.3GB 的 SAM 3 权重，
    所以默认跳过；正式验收前可单独打开。
    """

    def test_depth_anything_vitl_outputs_finite_nonconstant_depth(self) -> None:
        config = load_config("configs/demo.yaml")
        frame = _sample_frame(width=320)
        estimator = DepthAnythingV2Estimator(
            repo_dir=config["paths"]["depth_repo_dir"],
            checkpoint_path=config["paths"]["depth_checkpoint"],
            encoder=config["model"]["depth_encoder"],
            input_size=322,
            device=config["runtime"]["device"],
            allow_fallback=False,
        )

        result = estimator.predict(frame)

        self.assertEqual(result.backend, "official")
        self.assertEqual(result.depth.shape, frame.shape[:2])
        self.assertTrue(np.isfinite(result.depth).all())
        self.assertGreater(float(np.std(result.depth)), 1e-4)

    def test_sam3_outputs_valid_label_mask_contract(self) -> None:
        config = load_config("configs/demo.yaml")
        frame = _sample_frame(width=320)
        classes = config["model"]["sam3_classes"][:2]
        detector = SemanticDetector(
            model_path=config["paths"]["sam3_checkpoint"],
            classes=classes,
            conf=float(config["model"]["sam3_conf"]),
            imgsz=640,
            device=config["runtime"]["device"],
            half=bool(config["model"]["sam3_half"]),
            enabled=True,
        )

        result = detector.predict(frame)

        self.assertEqual(result.backend, "sam3")
        self.assertEqual(result.label_mask.shape, frame.shape[:2])
        self.assertTrue(np.issubdtype(result.label_mask.dtype, np.integer))
        self.assertGreaterEqual(int(result.label_mask.min()), 0)
        self.assertLessEqual(int(result.label_mask.max()), len(classes))
        for detection in result.detections:
            self.assertIn(detection.class_name, classes)
            self.assertGreaterEqual(detection.confidence, 0.0)
            self.assertLessEqual(detection.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
