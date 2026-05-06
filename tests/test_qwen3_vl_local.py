from __future__ import annotations

import unittest

from src.qwen3_vl_local import DEFAULT_SCENE_PROMPT, build_scene_prompt


class Qwen3VLLocalTest(unittest.TestCase):
    """不加载大模型，只验证本地 VLM 提示词构造。"""

    def test_build_scene_prompt_includes_metrics(self) -> None:
        metrics = {"nearest": {"class_name": "person", "median_z": 6.9}, "center_object_count": 2}

        prompt = build_scene_prompt(metrics)

        self.assertIn(DEFAULT_SCENE_PROMPT.splitlines()[0], prompt)
        self.assertIn('"class_name": "person"', prompt)
        self.assertIn('"center_object_count": 2', prompt)

    def test_custom_prompt_is_preserved(self) -> None:
        prompt = build_scene_prompt({"risk": "low"}, "请只输出一句话。")

        self.assertTrue(prompt.startswith("请只输出一句话。"))
        self.assertIn('"risk": "low"', prompt)


if __name__ == "__main__":
    unittest.main()
