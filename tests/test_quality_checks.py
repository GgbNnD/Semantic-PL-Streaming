from __future__ import annotations

import unittest

import numpy as np

from src.quality_checks import depth_quality_metrics, semantic_segmentation_metrics


class DepthQualityMetricsTest(unittest.TestCase):
    """验证深度估计正确性指标。"""

    def test_scaled_relative_depth_scores_as_perfect_after_alignment(self) -> None:
        reference = np.array(
            [
                [1.0, 2.0, 4.0],
                [2.0, 4.0, 8.0],
            ],
            dtype=np.float32,
        )
        predicted = reference * 7.0

        metrics = depth_quality_metrics(predicted, reference, align_median=True)

        self.assertAlmostEqual(metrics.abs_rel, 0.0, places=6)
        self.assertAlmostEqual(metrics.rmse, 0.0, places=6)
        self.assertAlmostEqual(metrics.delta1, 1.0, places=6)
        self.assertAlmostEqual(metrics.spearman, 1.0, places=6)
        self.assertEqual(metrics.valid_pixels, 6)
        self.assertAlmostEqual(metrics.scale, 1.0 / 7.0, places=6)

    def test_inverted_depth_has_bad_rank_agreement(self) -> None:
        reference = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        predicted = reference.max() + 1.0 - reference

        metrics = depth_quality_metrics(predicted, reference, align_median=True)

        self.assertLess(metrics.spearman, -0.99)
        self.assertLess(metrics.delta1, 0.5)

    def test_invalid_depth_pixels_are_ignored(self) -> None:
        reference = np.array([[1.0, 2.0], [0.0, np.nan]], dtype=np.float32)
        predicted = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)

        metrics = depth_quality_metrics(predicted, reference)

        self.assertEqual(metrics.valid_pixels, 2)
        self.assertAlmostEqual(metrics.abs_rel, 0.0, places=6)


class SemanticSegmentationMetricsTest(unittest.TestCase):
    """验证语义分割正确性指标。"""

    def test_perfect_mask_scores_as_one(self) -> None:
        reference = np.array(
            [
                [0, 1, 1],
                [0, 2, 2],
            ],
            dtype=np.int32,
        )

        metrics = semantic_segmentation_metrics(reference, reference, class_ids=[1, 2])

        self.assertAlmostEqual(metrics.mean_iou, 1.0, places=6)
        self.assertAlmostEqual(metrics.mean_dice, 1.0, places=6)
        self.assertAlmostEqual(metrics.pixel_accuracy, 1.0, places=6)
        self.assertEqual(metrics.per_class[1].support, 2)
        self.assertEqual(metrics.per_class[2].support, 2)

    def test_partial_mask_reports_expected_iou_and_recall(self) -> None:
        reference = np.array(
            [
                [0, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.int32,
        )
        predicted = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )

        metrics = semantic_segmentation_metrics(predicted, reference, class_ids=[1])
        class_one = metrics.per_class[1]

        # class 1: TP=2, FP=1, FN=1, IoU=2/4, Dice=4/6。
        self.assertAlmostEqual(class_one.iou, 0.5, places=6)
        self.assertAlmostEqual(class_one.dice, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(class_one.precision, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(class_one.recall, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics.mean_iou, 0.5, places=6)

    def test_ignore_index_excludes_unlabeled_pixels(self) -> None:
        reference = np.array([[1, 255], [0, 1]], dtype=np.int32)
        predicted = np.array([[1, 0], [0, 0]], dtype=np.int32)

        metrics = semantic_segmentation_metrics(predicted, reference, class_ids=[1], ignore_index=255)

        self.assertEqual(metrics.valid_pixels, 3)
        self.assertAlmostEqual(metrics.pixel_accuracy, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics.per_class[1].recall, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
