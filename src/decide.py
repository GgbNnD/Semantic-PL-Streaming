from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, resolve_path
from .decision import LocalRuleDecisionClient


def load_metrics(path: Path | None) -> dict:
    """读取 run_demo 保存的场景指标；缺失时返回空场景。"""

    if path is None or not path.exists():
        return {"nearest": None, "detections": [], "note": "未提供场景指标，按低风险输出。"}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于本地规则生成避障/风险建议")
    parser.add_argument("--config", default="configs/demo.yaml", help="配置文件路径")
    parser.add_argument("--snapshot", default=None, help="关键帧截图路径，仅用于报告记录")
    parser.add_argument("--metrics", default="outputs/decision/latest_scene_metrics.json", help="run_demo 生成的指标 JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = load_metrics(resolve_path(args.metrics) if args.metrics else None)
    client = LocalRuleDecisionClient(
        danger_z=float(config["decision"]["danger_z"]),
        warning_z=float(config["decision"]["warning_z"]),
    )
    text = client.analyze(metrics, resolve_path(args.snapshot) if args.snapshot else None)
    print(text)


if __name__ == "__main__":
    main()
