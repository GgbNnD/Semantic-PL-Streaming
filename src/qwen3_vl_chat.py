from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, resolve_path
from .qwen3_vl_local import Qwen3VLLocalClient, build_scene_prompt


def load_metrics(path: str | Path | None) -> dict | None:
    if not path:
        return None
    metrics_path = resolve_path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"指标文件不存在：{metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用本地 Qwen3-VL 分析图片/关键帧")
    parser.add_argument("--config", default="configs/demo.yaml", help="配置文件路径")
    parser.add_argument("--image", default="outputs/keyframes/frame_0090.jpg", help="输入图片路径")
    parser.add_argument("--metrics", default="outputs/decision/latest_scene_metrics.json", help="可选结构化指标 JSON")
    parser.add_argument("--prompt", default=None, help="自定义中文提示词")
    parser.add_argument("--model-dir", default=None, help="本地 Qwen3-VL 模型目录")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数")
    parser.add_argument("--max-memory-gb", type=float, default=None, help="GPU 最大显存预算")
    parser.add_argument("--output", default="outputs/decision/qwen3_vl_report.txt", help="保存输出文本")
    args = parser.parse_args()

    config = load_config(args.config)
    vlm_cfg = config["vlm"]
    model_dir = args.model_dir or vlm_cfg["qwen3_vl_local_dir"]
    max_new_tokens = args.max_new_tokens or int(vlm_cfg["qwen3_vl_max_new_tokens"])
    max_memory_gb = args.max_memory_gb or float(vlm_cfg["qwen3_vl_max_memory_gb"])

    metrics = load_metrics(args.metrics)
    prompt = build_scene_prompt(metrics, args.prompt)
    client = Qwen3VLLocalClient(model_dir=model_dir, max_memory_gb=max_memory_gb)
    answer = client.ask_image(args.image, prompt, max_new_tokens=max_new_tokens)

    print(answer)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(answer)
        f.write("\n")
    print(f"\n已保存：{output_path}")


if __name__ == "__main__":
    main()
