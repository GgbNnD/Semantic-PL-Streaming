from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import resolve_path


DEFAULT_SCENE_PROMPT = """请根据这张语义 3D 重建演示图和结构化指标，给出中文避障决策。
要求：
1. 先判断画面中主要目标、路面/可通行区域和潜在障碍物。
2. 结合伪距离 Z、最近目标和中心通道信息判断风险等级。
3. 明确说明当前深度是 Pseudo-LiDAR 相对伪距离，不是真实米制距离。
4. 输出 3-5 行，适合放进课程演示报告。"""


class Qwen3VLLocalClient:
    """本地 Qwen3-VL 推理封装。"""

    def __init__(
        self,
        model_dir: str | Path,
        max_memory_gb: float = 6.0,
        dtype: str = "auto",
    ) -> None:
        self.model_dir = resolve_path(model_dir)
        self.max_memory_gb = max_memory_gb
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Qwen3-VL 模型目录不存在：{self.model_dir}。请先运行 python -m src.prepare_qwen3_vl")

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        max_memory = None
        if torch.cuda.is_available():
            max_memory = {0: f"{self.max_memory_gb}GiB", "cpu": "24GiB"}

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            dtype=self.dtype,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))

    def ask_image(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        self.load()
        image_path = resolve_path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在：{image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output[0].strip() if output else ""


def build_scene_prompt(metrics: dict[str, Any] | None = None, user_prompt: str | None = None) -> str:
    prompt = user_prompt or DEFAULT_SCENE_PROMPT
    if not metrics:
        return prompt
    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    return f"{prompt}\n\n结构化指标如下：\n```json\n{metrics_text}\n```"
