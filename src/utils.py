from __future__ import annotations

import csv
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests

from .config import resolve_path


def now_ms() -> float:
    """返回当前时间戳，单位为毫秒，便于做模块级耗时统计。"""

    return time.perf_counter() * 1000.0


def run_command(command: list[str], cwd: str | Path | None = None) -> None:
    """执行外部命令，并在失败时给出完整命令，方便定位环境问题。"""

    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(f"命令执行失败：{joined}")


def download_file(url: str, target: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    """流式下载文件，避免一次性把模型权重读入内存。"""

    target = resolve_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".part")

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100.0
                    print(f"下载 {target.name}: {percent:5.1f}%", end="\r")
    tmp_path.replace(target)
    print(f"下载完成：{target}")
    return target


def copy_if_newer(source: str | Path, target: str | Path) -> None:
    """把文件复制到目标位置；目标已存在时不重复覆盖。"""

    source = resolve_path(source)
    target = resolve_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def resize_keep_aspect(frame_bgr: np.ndarray, target_width: int | None) -> np.ndarray:
    """按指定宽度缩放图像，保持宽高比。

    深度模型和检测模型都可以处理原图，但课程演示更关注管线结构和 CUDA 加速，
    因此默认把较大的 1080P 视频缩到 960 宽，降低显存和渲染压力。
    """

    if not target_width or frame_bgr.shape[1] <= target_width:
        return frame_bgr
    scale = target_width / frame_bgr.shape[1]
    target_size = (target_width, int(round(frame_bgr.shape[0] * scale)))
    return cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """把任意浮点数组归一化到 0-255，用于深度图热力显示。"""

    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    v_min = float(values[finite].min())
    v_max = float(values[finite].max())
    if math.isclose(v_min, v_max):
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - v_min) / (v_max - v_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def relative_depth_to_pseudo_z(
    depth: np.ndarray,
    min_z: float,
    max_z: float,
    invert: bool = True,
) -> np.ndarray:
    """把相对深度映射成伪 3D 距离 Z。

    Depth-Anything-V2 的通用模型输出的是相对/逆深度，不是米制距离。为了展示
    Pseudo-LiDAR 点云，我们先把模型输出归一化到 0-1，再映射到 `[min_z, max_z]`。
    当 `invert=True` 时，较大的网络输出被解释为“更近”，对应更小的 Z。
    """

    depth_u8 = normalize_to_uint8(depth).astype(np.float32) / 255.0
    if invert:
        depth_u8 = 1.0 - depth_u8
    z = min_z + depth_u8 * (max_z - min_z)
    return z.astype(np.float32)


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    """写出 CSV；字段名自动从第一行推断。"""

    rows = list(rows)
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_video_writer(path: str | Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    """创建 OpenCV 视频写入器，默认使用 mp4v 编码。"""

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频输出：{path}")
    return writer


def image_grid(images: list[np.ndarray], cols: int = 2, cell_width: int = 640) -> np.ndarray:
    """把若干 BGR 图像拼成固定网格，作为最终演示视频的一帧。"""

    if not images:
        raise ValueError("image_grid 至少需要一张图像")
    resized: list[np.ndarray] = []
    for image in images:
        h, w = image.shape[:2]
        scale = cell_width / max(w, 1)
        cell = cv2.resize(image, (cell_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        resized.append(cell)

    cell_height = max(img.shape[0] for img in resized)
    padded: list[np.ndarray] = []
    for image in resized:
        if image.shape[0] < cell_height:
            pad = np.zeros((cell_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
            image = np.vstack([image, pad])
        padded.append(image)

    rows: list[np.ndarray] = []
    for start in range(0, len(padded), cols):
        row_images = padded[start : start + cols]
        while len(row_images) < cols:
            row_images.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row_images))
    return np.vstack(rows)
