from __future__ import annotations

import unittest

import numpy as np
import torch

from src.semantic_detector import SemanticDetector, _extract_masks, _version_at_least


class SemanticDetectorTest(unittest.TestCase):
    """验证 SAM 3 语义封装中不依赖大模型的基础逻辑。"""

    def test_disabled_detector_returns_blank_mask(self) -> None:
        detector = SemanticDetector(
            model_path="third_party/sam3/sam3.pt",
            classes=["person"],
            enabled=False,
        )
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        result = detector.predict(frame)

        self.assertEqual(result.backend, "disabled")
        self.assertEqual(result.label_mask.shape, (6, 8))
        self.assertFalse(result.label_mask.any())

    def test_version_check_accepts_sam3_release(self) -> None:
        self.assertTrue(_version_at_least("8.3.237", (8, 3, 237)))
        self.assertTrue(_version_at_least("8.4.41", (8, 3, 237)))
        self.assertFalse(_version_at_least("8.3.120", (8, 3, 237)))

    def test_extract_masks_resizes_to_frame_shape(self) -> None:
        class FakeMasks:
            data = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)

        class FakeResult:
            masks = FakeMasks()

        masks = _extract_masks(FakeResult(), (4, 4))

        self.assertIsNotNone(masks)
        self.assertEqual(masks.shape, (1, 4, 4))
        self.assertTrue(masks[0, 0, 0])
        self.assertTrue(masks[0, 3, 3])


if __name__ == "__main__":
    unittest.main()
