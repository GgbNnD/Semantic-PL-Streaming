from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, resolve_path


def download_with_modelscope(model_id: str, local_dir: Path) -> str:
    """使用 ModelScope 下载 Qwen3-VL 权重。"""

    from modelscope import snapshot_download

    return snapshot_download(
        model_id=model_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.bin", "*.onnx", "*.tflite"],
        max_workers=4,
    )


def download_with_huggingface(model_id: str, local_dir: Path) -> str:
    """使用 Hugging Face Hub 下载 Qwen3-VL 权重。"""

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.bin", "*.onnx", "*.tflite"],
        max_workers=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载本地 Qwen3-VL 模型")
    parser.add_argument("--config", default="configs/demo.yaml", help="配置文件路径")
    parser.add_argument("--model-id", default=None, help="Hugging Face/ModelScope 模型 ID")
    parser.add_argument("--local-dir", default=None, help="模型保存目录")
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope", help="下载源")
    args = parser.parse_args()

    config = load_config(args.config)
    vlm_cfg = config["vlm"]
    model_id = args.model_id or vlm_cfg["qwen3_vl_model_id"]
    local_dir = resolve_path(args.local_dir or vlm_cfg["qwen3_vl_local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型选择：{model_id}")
    print(f"保存目录：{local_dir}")
    if args.source == "modelscope":
        path = download_with_modelscope(model_id, local_dir)
    else:
        path = download_with_huggingface(model_id, local_dir)
    print(f"Qwen3-VL 模型准备完成：{path}")


if __name__ == "__main__":
    main()
