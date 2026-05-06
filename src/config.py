from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "third_party_dir": "third_party",
        "depth_repo_dir": "third_party/Depth-Anything-V2",
        "depth_checkpoint": "models/depth_anything_v2_vitl.pth",
        "sam3_checkpoint": "third_party/sam3/sam3.pt",
        "input_video": "data/input/demo.mp4",
        "output_dir": "outputs",
        "benchmark_dir": "benchmarks",
    },
    "model": {
        "depth_encoder": "vitl",
        "depth_input_size": 518,
        "sam3_imgsz": 644,
        "sam3_conf": 0.25,
        "sam3_half": True,
        "sam3_classes": [
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "traffic cone",
            "obstacle",
        ],
    },
    "projection": {
        "stride": 4,
        "fov_degrees": 70.0,
        "pseudo_min_z": 0.5,
        "pseudo_max_z": 20.0,
        "invert_depth": True,
        "filter_min_z": 0.3,
        "filter_max_z": 25.0,
    },
    "filtering": {
        "enabled": True,
        "radius": 1,
        "depth_jump_threshold": 1.2,
        "min_neighbors": 3,
    },
    "runtime": {
        "device": "cuda",
        "max_frames": 120,
        "output_fps": 15,
        "keyframe_interval": 30,
        "process_width": 960,
        "use_sam3": True,
        "allow_fallback_depth": True,
        "save_ply_interval": 60,
    },
    "visualization": {
        "live_open3d": False,
    },
    "evaluation": {
        "enabled": True,
    },
    "decision": {
        "center_band_ratio": 0.40,
        "danger_z": 3.0,
        "warning_z": 6.0,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典。

    课程项目的配置项比较少，但拆成多个模块后仍然需要一个稳定的合并逻辑。
    这里采用“用户配置覆盖默认配置”的策略，避免因为 YAML 少写某个字段就让程序
    缺失默认值。
    """

    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path = "configs/demo.yaml") -> dict[str, Any]:
    """读取 YAML 配置，并补齐默认值。"""

    path = resolve_path(config_path)
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def resolve_path(path: str | Path) -> Path:
    """把配置里的相对路径解析到项目根目录下。

    所有命令都从 `/home/cells/ai` 这个项目根目录运行更直观，但为了允许用户在
    其他目录调用 `python -m src.run_demo`，内部统一把相对路径转成绝对路径。
    """

    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_project_dirs(config: dict[str, Any]) -> None:
    """创建项目运行所需目录。"""

    paths = config["paths"]
    for key in [
        "third_party_dir",
        "depth_repo_dir",
        "depth_checkpoint",
        "sam3_checkpoint",
        "input_video",
        "output_dir",
        "benchmark_dir",
    ]:
        target = resolve_path(paths[key])
        # checkpoint 和 input_video 是文件路径，创建它们的父目录即可。
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
