from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .semantic_detector import Detection


@dataclass
class DetectionMetric:
    """带伪深度统计的检测目标。"""

    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    median_z: float
    center_overlap: bool


def compute_scene_metrics(
    z_map: np.ndarray,
    detections: list[Detection],
    center_band_ratio: float,
) -> dict[str, Any]:
    """根据语义实例外接框和伪深度图计算本地决策所需指标。

    由于当前没有 VLM API，本地规则重点关注三个量：最近障碍物、它是否落在画面
    中心通道、以及它的伪距离 Z。Z 越小表示越近。
    """

    h, w = z_map.shape
    band_w = int(w * center_band_ratio)
    band_left = (w - band_w) // 2
    band_right = band_left + band_w
    metrics: list[DetectionMetric] = []

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        crop = z_map[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            continue
        finite = crop[np.isfinite(crop)]
        if finite.size == 0:
            continue
        center_x = (x1 + x2) * 0.5
        center_overlap = band_left <= center_x <= band_right
        metrics.append(
            DetectionMetric(
                class_name=det.class_name,
                confidence=det.confidence,
                xyxy=det.xyxy,
                median_z=float(np.median(finite)),
                center_overlap=center_overlap,
            )
        )

    nearest = min(metrics, key=lambda item: item.median_z, default=None)
    center_objects = [item for item in metrics if item.center_overlap]
    return {
        "detections": [asdict(item) for item in metrics],
        "nearest": asdict(nearest) if nearest else None,
        "center_object_count": len(center_objects),
        "z_min": float(np.nanmin(z_map)) if z_map.size else None,
        "z_p10": float(np.nanpercentile(z_map, 10)) if z_map.size else None,
        "center_band": {"left": band_left, "right": band_right},
    }


class DecisionClient:
    """决策接口基类。

    以后接入 Qwen-VL 时，只需要新增一个云端实现并保持 `analyze` 返回中文文本即可。
    """

    def analyze(self, metrics: dict[str, Any], snapshot_path: str | Path | None = None) -> str:
        raise NotImplementedError


class LocalRuleDecisionClient(DecisionClient):
    """无 API 时使用的本地规则决策器。"""

    def __init__(self, danger_z: float, warning_z: float) -> None:
        self.danger_z = danger_z
        self.warning_z = warning_z

    def analyze(self, metrics: dict[str, Any], snapshot_path: str | Path | None = None) -> str:
        nearest = metrics.get("nearest")
        if nearest is None:
            return "风险等级：低。\n未检测到明确语义障碍物，建议保持观察并继续直行。"

        z = float(nearest["median_z"])
        name = nearest["class_name"]
        in_center = bool(nearest["center_overlap"])
        if z <= self.danger_z and in_center:
            level = "高"
            advice = "立即减速或停车，并向左右可通行区域重新规划路径。"
        elif z <= self.warning_z and in_center:
            level = "中"
            advice = "前方中心通道存在较近目标，建议提前减速并准备绕行。"
        elif z <= self.warning_z:
            level = "中低"
            advice = "侧前方存在较近目标，建议保持车道中心并持续观察。"
        else:
            level = "低"
            advice = "障碍物距离相对较远，维持当前速度并持续监测即可。"

        return (
            f"风险等级：{level}。\n"
            f"最近目标：{name}，伪距离 Z≈{z:.2f}，"
            f"{'位于画面中心通道' if in_center else '未位于画面中心通道'}。\n"
            f"建议：{advice}\n"
            "说明：当前结果基于相对深度 Pseudo-LiDAR 和本地规则，尚未调用云端 VLM。"
        )


class CloudVLMDecisionClient(DecisionClient):
    """Qwen-VL 云端接口占位实现。

    现在用户还没有 API key，因此不主动调用网络。后续只要设置 `DASHSCOPE_API_KEY`
    并补充真实请求逻辑，就可以替换本地规则输出。
    """

    def analyze(self, metrics: dict[str, Any], snapshot_path: str | Path | None = None) -> str:
        if not os.getenv("DASHSCOPE_API_KEY"):
            return "未检测到 DASHSCOPE_API_KEY，当前保持本地规则决策。"
        return "DASHSCOPE_API_KEY 已存在，但云端 VLM 调用逻辑在无 API 版本中保持关闭。"


def save_decision_artifacts(
    metrics: dict[str, Any],
    decision_text: str,
    output_dir: str | Path,
) -> None:
    """保存本地决策的 JSON 指标和中文文本报告。"""

    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "latest_scene_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "decision_report.txt").open("w", encoding="utf-8") as f:
        f.write(decision_text)
        f.write("\n")
