from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthQualityMetrics:
    """深度估计质量指标。

    Depth-Anything-V2 通用模型输出相对深度，因此默认先用中位数做尺度对齐，
    再计算误差；Spearman 用来衡量近远排序是否一致。
    """

    abs_rel: float
    rmse: float
    delta1: float
    spearman: float
    valid_pixels: int
    scale: float


@dataclass(frozen=True)
class ClassSegmentationMetrics:
    """单个语义类别的 mask 质量指标。"""

    class_id: int
    iou: float
    dice: float
    precision: float
    recall: float
    support: int


@dataclass(frozen=True)
class SemanticSegmentationMetrics:
    """语义分割整体质量指标。"""

    mean_iou: float
    mean_dice: float
    pixel_accuracy: float
    valid_pixels: int
    per_class: dict[int, ClassSegmentationMetrics]


def depth_quality_metrics(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    align_median: bool = True,
    eps: float = 1e-6,
) -> DepthQualityMetrics:
    """计算深度估计质量指标。

    Args:
        predicted_depth: 模型输出深度，形状与参考深度一致。
        reference_depth: 参考/标注深度。可以是真实米制深度，也可以是相对深度。
        valid_mask: 可选有效像素 mask。
        align_median: 是否用参考深度和预测深度的中位数做尺度对齐。
        eps: 避免除零的最小正数。
    """

    pred = np.asarray(predicted_depth, dtype=np.float32)
    ref = np.asarray(reference_depth, dtype=np.float32)
    if pred.shape != ref.shape:
        raise ValueError(f"predicted_depth 和 reference_depth 形状不一致：{pred.shape} vs {ref.shape}")

    mask = np.isfinite(pred) & np.isfinite(ref) & (pred > eps) & (ref > eps)
    if valid_mask is not None:
        if valid_mask.shape != pred.shape:
            raise ValueError(f"valid_mask 形状不一致：{valid_mask.shape} vs {pred.shape}")
        mask &= valid_mask.astype(bool)

    pred_values = pred[mask]
    ref_values = ref[mask]
    if pred_values.size == 0:
        raise ValueError("没有有效深度像素可用于评估")

    scale = 1.0
    if align_median:
        pred_median = float(np.median(pred_values))
        ref_median = float(np.median(ref_values))
        if pred_median > eps:
            scale = ref_median / pred_median

    pred_aligned = np.maximum(pred_values * scale, eps)
    abs_rel = float(np.mean(np.abs(pred_aligned - ref_values) / ref_values))
    rmse = float(np.sqrt(np.mean((pred_aligned - ref_values) ** 2)))
    ratio = np.maximum(pred_aligned / ref_values, ref_values / pred_aligned)
    delta1 = float(np.mean(ratio < 1.25))
    spearman = _spearman(pred_values, ref_values)

    return DepthQualityMetrics(
        abs_rel=abs_rel,
        rmse=rmse,
        delta1=delta1,
        spearman=spearman,
        valid_pixels=int(pred_values.size),
        scale=float(scale),
    )


def semantic_segmentation_metrics(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
    class_ids: list[int] | tuple[int, ...] | None = None,
    ignore_index: int | None = None,
) -> SemanticSegmentationMetrics:
    """计算语义分割 mask 的 IoU/Dice/Precision/Recall。

    `0` 默认视为背景；如果没有显式传入 `class_ids`，函数会评估预测和标注中
    出现过的所有非背景类别。
    """

    pred = np.asarray(predicted_mask)
    ref = np.asarray(reference_mask)
    if pred.shape != ref.shape:
        raise ValueError(f"predicted_mask 和 reference_mask 形状不一致：{pred.shape} vs {ref.shape}")

    valid = np.ones(ref.shape, dtype=bool)
    if ignore_index is not None:
        valid &= ref != ignore_index

    if class_ids is None:
        labels = set(np.unique(pred[valid]).astype(int).tolist())
        labels.update(np.unique(ref[valid]).astype(int).tolist())
        labels.discard(0)
        if ignore_index is not None:
            labels.discard(int(ignore_index))
        class_ids = sorted(labels)

    per_class: dict[int, ClassSegmentationMetrics] = {}
    for class_id in class_ids:
        label = int(class_id)
        pred_pos = (pred == label) & valid
        ref_pos = (ref == label) & valid
        tp = int(np.count_nonzero(pred_pos & ref_pos))
        fp = int(np.count_nonzero(pred_pos & ~ref_pos))
        fn = int(np.count_nonzero(~pred_pos & ref_pos))
        support = int(np.count_nonzero(ref_pos))

        iou_den = tp + fp + fn
        dice_den = 2 * tp + fp + fn
        precision_den = tp + fp
        recall_den = tp + fn
        per_class[label] = ClassSegmentationMetrics(
            class_id=label,
            iou=float(tp / iou_den) if iou_den else 1.0,
            dice=float((2 * tp) / dice_den) if dice_den else 1.0,
            precision=float(tp / precision_den) if precision_den else 1.0,
            recall=float(tp / recall_den) if recall_den else 1.0,
            support=support,
        )

    valid_pixels = int(np.count_nonzero(valid))
    pixel_accuracy = float(np.mean(pred[valid] == ref[valid])) if valid_pixels else 0.0
    class_metrics = list(per_class.values())
    mean_iou = float(np.mean([item.iou for item in class_metrics])) if class_metrics else 1.0
    mean_dice = float(np.mean([item.dice for item in class_metrics])) if class_metrics else 1.0
    return SemanticSegmentationMetrics(
        mean_iou=mean_iou,
        mean_dice=mean_dice,
        pixel_accuracy=pixel_accuracy,
        valid_pixels=valid_pixels,
        per_class=per_class,
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a.astype(np.float64) - float(np.mean(a))
    b_centered = b.astype(np.float64) - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a_centered * b_centered) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) * 0.5 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks
