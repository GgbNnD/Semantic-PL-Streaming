from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from .config import load_config, resolve_path
from .decision import LocalRuleDecisionClient, compute_scene_metrics
from .depth_estimator import DepthAnythingV2Estimator
from .filters import DepthFilter
from .qwen3_vl_local import Qwen3VLLocalClient
from .semantic_detector import SemanticDetector
from .utils import now_ms, relative_depth_to_pseudo_z, resize_keep_aspect


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>语义 3D 决策界面</title>
  <style>
    :root {
      --bg: #f4f5f2;
      --ink: #20231f;
      --muted: #62695f;
      --line: #d7dacf;
      --panel: #ffffff;
      --panel-2: #eef2e8;
      --green: #2e7d55;
      --amber: #ad6b00;
      --red: #b13d34;
      --blue: #316a86;
      --shadow: 0 10px 24px rgba(32, 35, 31, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(8px);
      padding: 14px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }

    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    select, input, button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
    }

    select {
      min-width: min(380px, 54vw);
      padding: 0 12px;
    }

    input {
      width: 88px;
      padding: 0 10px;
    }

    button {
      padding: 0 14px;
      cursor: pointer;
      font-weight: 650;
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }

    button:hover { transform: translateY(-1px); }

    .primary {
      background: var(--green);
      border-color: var(--green);
      color: #fff;
    }

    .secondary {
      background: #fff;
      border-color: #bfc7b8;
    }

    main {
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(420px, 1.45fr) minmax(340px, 0.8fr);
      gap: 18px;
      min-height: 0;
    }

    section {
      min-width: 0;
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .panel-head {
      min-height: 46px;
      border-bottom: 1px solid var(--line);
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: var(--panel-2);
    }

    .panel-head h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 2px 10px;
      font-size: 13px;
      color: var(--muted);
      background: #fff;
      white-space: nowrap;
    }

    .stage {
      flex: 1;
      min-height: 0;
      background: #161913;
      display: grid;
      place-items: center;
      padding: 12px;
    }

    #videoFrame {
      width: 100%;
      max-height: calc(100vh - 148px);
      object-fit: contain;
      border-radius: 6px;
      background: #10120f;
    }

    .decision-body {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow: auto;
    }

    .risk-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      min-height: 74px;
    }

    .metric label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
      margin-bottom: 6px;
    }

    .metric strong {
      display: block;
      font-size: clamp(18px, 2vw, 24px);
      line-height: 1.15;
      overflow-wrap: anywhere;
    }

    .risk-low strong { color: var(--green); }
    .risk-mid strong { color: var(--amber); }
    .risk-high strong { color: var(--red); }

    .decision-text {
      border-left: 4px solid var(--blue);
      background: #f7f9f5;
      padding: 12px;
      border-radius: 6px;
      white-space: pre-wrap;
      line-height: 1.58;
      font-size: 15px;
      min-height: 116px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px 6px;
      vertical-align: top;
    }

    th {
      color: var(--muted);
      font-weight: 650;
      background: #fafbf8;
    }

    .kv {
      display: grid;
      grid-template-columns: 112px 1fr;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }

    .kv span:nth-child(even) {
      color: var(--ink);
      overflow-wrap: anywhere;
    }

    @media (max-width: 920px) {
      header {
        align-items: stretch;
        flex-direction: column;
      }

      .toolbar {
        justify-content: flex-start;
      }

      select {
        min-width: 100%;
      }

      main {
        grid-template-columns: 1fr;
      }

      #videoFrame {
        max-height: 54vh;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>语义级 3D 场景重建与实时决策</h1>
      <div class="toolbar">
        <select id="videoSelect" aria-label="视频源"></select>
        <select id="backendSelect" aria-label="决策后端" title="决策后端"></select>
        <input id="maxFrames" type="number" min="0" step="1" aria-label="帧数" title="0 表示处理到视频结束">
        <input id="stride" type="number" min="1" step="1" aria-label="间隔">
        <input id="delay" type="number" min="0" step="50" aria-label="延迟" title="每帧延迟，单位毫秒">
        <button class="primary" id="startBtn">启动</button>
        <button class="secondary" id="stopBtn">停止</button>
      </div>
    </header>

    <main>
      <section>
        <div class="panel-head">
          <h2>原始视频</h2>
          <span class="pill" id="videoName">未选择</span>
        </div>
        <div class="stage">
          <img id="videoFrame" src="/stream" alt="原始视频帧">
        </div>
      </section>

      <section>
        <div class="panel-head">
          <h2>实时决策</h2>
          <span class="pill" id="statePill">idle</span>
        </div>
        <div class="decision-body">
          <div class="risk-row">
            <div class="metric" id="riskMetric">
              <label>风险等级</label>
              <strong id="riskLevel">-</strong>
            </div>
            <div class="metric">
              <label>最近目标</label>
              <strong id="nearestTarget">-</strong>
            </div>
            <div class="metric">
              <label>伪距离 Z</label>
              <strong id="nearestZ">-</strong>
            </div>
            <div class="metric">
              <label>当前帧</label>
              <strong id="frameIndex">0</strong>
            </div>
          </div>

          <div class="decision-text" id="decisionText">等待启动。</div>

          <div class="kv">
            <span>深度后端</span><span id="depthBackend">-</span>
            <span>语义后端</span><span id="semanticBackend">-</span>
            <span>滤波后端</span><span id="filterBackend">-</span>
            <span>决策后端</span><span id="decisionBackend">-</span>
            <span>单帧耗时</span><span id="latency">-</span>
          </div>

          <table>
            <thead>
              <tr>
                <th>目标</th>
                <th>置信度</th>
                <th>Z</th>
                <th>中心</th>
              </tr>
            </thead>
            <tbody id="detections"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <script>
    const videoSelect = document.getElementById("videoSelect");
    const backendSelect = document.getElementById("backendSelect");
    const maxFrames = document.getElementById("maxFrames");
    const stride = document.getElementById("stride");
    const delay = document.getElementById("delay");
    const statePill = document.getElementById("statePill");
    const decisionText = document.getElementById("decisionText");
    const riskMetric = document.getElementById("riskMetric");

    async function loadVideos() {
      const res = await fetch("/api/videos");
      const data = await res.json();
      videoSelect.innerHTML = "";
      data.videos.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.path;
        option.textContent = item.name;
        videoSelect.appendChild(option);
      });
      backendSelect.innerHTML = "";
      data.defaults.backends.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.value;
        option.textContent = item.label;
        backendSelect.appendChild(option);
      });
      backendSelect.value = data.defaults.decision_backend;
      maxFrames.value = data.defaults.max_frames;
      stride.value = data.defaults.decision_stride;
      delay.value = data.defaults.frame_delay_ms;
      document.getElementById("videoName").textContent = videoSelect.value || "未选择";
    }

    async function startRun() {
      const body = {
        video: videoSelect.value,
        decision_backend: backendSelect.value,
        max_frames: Number(maxFrames.value || 0),
        decision_stride: Math.max(1, Number(stride.value || 1)),
        frame_delay_ms: Math.max(0, Number(delay.value || 0))
      };
      await fetch("/api/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      document.getElementById("videoName").textContent = videoSelect.value || "未选择";
      document.getElementById("videoFrame").src = "/stream?t=" + Date.now();
      refreshStatus();
    }

    async function stopRun() {
      await fetch("/api/stop", {method: "POST"});
      refreshStatus();
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value ?? "-";
    }

    function updateRiskClass(risk) {
      riskMetric.classList.remove("risk-low", "risk-mid", "risk-high");
      if (!risk || risk === "-") return;
      if (risk.includes("高")) riskMetric.classList.add("risk-high");
      else if (risk.includes("中")) riskMetric.classList.add("risk-mid");
      else riskMetric.classList.add("risk-low");
    }

    function renderDetections(items) {
      const tbody = document.getElementById("detections");
      tbody.innerHTML = "";
      if (!items || items.length === 0) {
        const row = document.createElement("tr");
        row.innerHTML = "<td colspan='4'>-</td>";
        tbody.appendChild(row);
        return;
      }
      items.slice(0, 8).forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${item.class_name}</td>
          <td>${Number(item.confidence).toFixed(2)}</td>
          <td>${Number(item.median_z).toFixed(2)}</td>
          <td>${item.center_overlap ? "是" : "否"}</td>
        `;
        tbody.appendChild(row);
      });
    }

    async function refreshStatus() {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        statePill.textContent = data.status;
        setText("riskLevel", data.risk_level || "-");
        setText("nearestTarget", data.nearest?.class_name || "-");
        const zText = data.nearest?.z_text || (data.nearest?.median_z != null ? Number(data.nearest.median_z).toFixed(2) : "-");
        setText("nearestZ", zText);
        setText("frameIndex", `${data.frame_index || 0}${data.total_frames ? " / " + data.total_frames : ""}`);
        setText("depthBackend", data.backends.depth || "-");
        setText("semanticBackend", data.backends.semantic || "-");
        setText("filterBackend", data.backends.filter || "-");
        setText("decisionBackend", data.backends.decision || "-");
        setText("latency", data.timings.total_ms ? `${Number(data.timings.total_ms).toFixed(1)} ms` : "-");
        decisionText.textContent = data.error || data.decision_text || "等待启动。";
        updateRiskClass(data.risk_level);
        renderDetections(data.detections);
      } catch (err) {
        statePill.textContent = "offline";
      }
    }

    document.getElementById("startBtn").addEventListener("click", startRun);
    document.getElementById("stopBtn").addEventListener("click", stopRun);
    videoSelect.addEventListener("change", () => {
      document.getElementById("videoName").textContent = videoSelect.value || "未选择";
    });

    loadVideos().then(refreshStatus);
    setInterval(refreshStatus, 800);
  </script>
</body>
</html>
"""


@dataclass
class UIStatus:
    status: str = "idle"
    selected_video: str = ""
    frame_index: int = 0
    total_frames: int = 0
    decision_text: str = "等待启动。"
    risk_level: str = "-"
    nearest: dict[str, Any] | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    backends: dict[str, str] = field(default_factory=dict)
    decision_backend: str = "qwen3_vl"
    error: str = ""
    updated_at: float = field(default_factory=time.time)


class DecisionUIState:
    def __init__(
        self,
        config: dict[str, Any],
        default_max_frames: int,
        default_decision_stride: int,
        default_decision_backend: str,
        default_frame_delay_ms: int,
    ) -> None:
        self.config = config
        self.default_max_frames = default_max_frames
        self.default_decision_stride = default_decision_stride
        self.default_decision_backend = default_decision_backend
        self.default_frame_delay_ms = default_frame_delay_ms
        self.lock = threading.Lock()
        self.status = UIStatus(decision_backend=default_decision_backend)
        self.frame_jpeg = placeholder_jpeg("No video")
        self.frame_version = 0
        self.worker: RealtimeDecisionWorker | None = None

    def start(
        self,
        video_path: Path,
        max_frames: int,
        decision_stride: int,
        decision_backend: str,
        frame_delay_ms: int,
    ) -> None:
        if not self.stop():
            raise RuntimeError("上一轮任务仍在停止中，请稍后再启动。")
        selected = str(video_path.relative_to(resolve_path("."))) if video_path.is_relative_to(resolve_path(".")) else str(video_path)
        with self.lock:
            self.status = UIStatus(
                status="loading",
                selected_video=selected,
                decision_backend=decision_backend,
                decision_text="模型加载中。",
                backends={"decision": decision_backend},
            )
            self.frame_jpeg = placeholder_jpeg("Loading")
            self.frame_version += 1
        self.worker = RealtimeDecisionWorker(
            self,
            self.config,
            video_path,
            max_frames,
            decision_stride,
            decision_backend,
            frame_delay_ms,
        )
        self.worker.start()

    def stop(self) -> bool:
        worker = self.worker
        if worker and worker.is_alive():
            worker.stop()
            worker.join(timeout=3)
            if worker.is_alive():
                with self.lock:
                    self.status.status = "stopping"
                    self.status.updated_at = time.time()
                return False
        self.worker = None
        with self.lock:
            if self.status.status in {"loading", "running"}:
                self.status.status = "stopped"
                self.status.updated_at = time.time()
        return True

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with self.lock:
            self.frame_jpeg = encoded.tobytes()
            self.frame_version += 1

    def update_status(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self.status, key, value)
            self.status.updated_at = time.time()

    def snapshot_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status.status,
                "selected_video": self.status.selected_video,
                "frame_index": self.status.frame_index,
                "total_frames": self.status.total_frames,
                "decision_text": self.status.decision_text,
                "risk_level": self.status.risk_level,
                "nearest": self.status.nearest,
                "detections": self.status.detections,
                "timings": self.status.timings,
                "backends": {
                    "depth": self.status.backends.get("depth", "-"),
                    "semantic": self.status.backends.get("semantic", "-"),
                    "filter": self.status.backends.get("filter", "-"),
                    "decision": self.status.backends.get("decision", self.status.decision_backend),
                },
                "decision_backend": self.status.decision_backend,
                "error": self.status.error,
                "updated_at": self.status.updated_at,
            }

    def snapshot_frame(self) -> tuple[bytes, int]:
        with self.lock:
            return self.frame_jpeg, self.frame_version


class RealtimeDecisionWorker(threading.Thread):
    def __init__(
        self,
        state: DecisionUIState,
        config: dict[str, Any],
        video_path: Path,
        max_frames: int,
        decision_stride: int,
        decision_backend: str,
        frame_delay_ms: int,
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.config = config
        self.video_path = video_path
        self.max_frames = max_frames
        self.decision_stride = max(1, decision_stride)
        self.decision_backend = decision_backend
        self.frame_delay_ms = max(0, frame_delay_ms)
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            runtime = self.config["runtime"]
            model_cfg = self.config["model"]
            proj_cfg = self.config["projection"]
            filter_cfg = self.config["filtering"]
            decision_cfg = self.config["decision"]
            vlm_cfg = self.config["vlm"]

            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"无法打开视频：{self.video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            depth_estimator = None
            semantic_detector = None
            depth_filter = None
            decision_client = None
            qwen_client = None
            qwen_frame_path = resolve_path(self.config["paths"]["output_dir"]) / "ui" / "qwen3_vl_current.jpg"
            qwen_frame_path.parent.mkdir(parents=True, exist_ok=True)

            if self.decision_backend == "local_rules":
                depth_estimator = DepthAnythingV2Estimator(
                    repo_dir=self.config["paths"]["depth_repo_dir"],
                    checkpoint_path=self.config["paths"]["depth_checkpoint"],
                    encoder=model_cfg["depth_encoder"],
                    input_size=int(model_cfg["depth_input_size"]),
                    device=runtime["device"],
                    allow_fallback=bool(runtime["allow_fallback_depth"]),
                )
                semantic_detector = SemanticDetector(
                    model_path=self.config["paths"].get("sam3_checkpoint", "third_party/sam3/sam3.pt"),
                    classes=model_cfg["sam3_classes"],
                    conf=float(model_cfg["sam3_conf"]),
                    imgsz=int(model_cfg["sam3_imgsz"]),
                    device=runtime["device"],
                    half=bool(model_cfg.get("sam3_half", True)),
                    enabled=bool(runtime["use_sam3"]),
                )
                depth_filter = DepthFilter(
                    enabled=bool(filter_cfg["enabled"]),
                    radius=int(filter_cfg["radius"]),
                    jump_threshold=float(filter_cfg["depth_jump_threshold"]),
                    min_neighbors=int(filter_cfg["min_neighbors"]),
                )
                decision_client = LocalRuleDecisionClient(
                    danger_z=float(decision_cfg["danger_z"]),
                    warning_z=float(decision_cfg["warning_z"]),
                )
            else:
                qwen_client = Qwen3VLLocalClient(
                    model_dir=vlm_cfg["qwen3_vl_local_dir"],
                    max_memory_gb=float(vlm_cfg["qwen3_vl_max_memory_gb"]),
                )

            frame_index = 0
            self.state.update_status(
                status="running",
                total_frames=total_frames,
                decision_backend=self.decision_backend,
                backends={"decision": self.decision_backend},
                error="",
            )

            while not self.stop_event.is_set():
                if self.max_frames > 0 and frame_index >= self.max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                loop_start = now_ms()
                frame = resize_keep_aspect(frame, int(runtime["process_width"]))
                self.state.update_frame(frame)

                if frame_index % self.decision_stride == 0:
                    if self.decision_backend == "local_rules":
                        assert depth_estimator is not None
                        assert semantic_detector is not None
                        assert depth_filter is not None
                        assert decision_client is not None
                        depth_result = depth_estimator.predict(frame)
                        z_map = relative_depth_to_pseudo_z(
                            depth_result.depth,
                            min_z=float(proj_cfg["pseudo_min_z"]),
                            max_z=float(proj_cfg["pseudo_max_z"]),
                            invert=bool(proj_cfg["invert_depth"]),
                        )
                        filter_result = depth_filter.filter_auto(z_map, prefer_cuda=runtime["device"] == "cuda")
                        semantic_result = semantic_detector.predict(frame)
                        metrics = compute_scene_metrics(
                            filter_result.z_map,
                            semantic_result.detections,
                            center_band_ratio=float(decision_cfg["center_band_ratio"]),
                        )
                        metrics["frame_index"] = frame_index
                        decision_text = decision_client.analyze(metrics)
                        nearest = metrics.get("nearest")
                        detections = metrics.get("detections", [])[:12]
                        risk_level = extract_risk_level(decision_text)
                        self.state.update_status(
                            status="running",
                            frame_index=frame_index,
                            decision_text=decision_text,
                            risk_level=risk_level,
                            nearest=nearest,
                            detections=detections,
                            timings={
                                "depth_ms": round(depth_result.elapsed_ms, 2),
                                "filter_ms": round(filter_result.elapsed_ms, 2),
                                "semantic_ms": round(semantic_result.elapsed_ms, 2),
                                "total_ms": round(now_ms() - loop_start, 2),
                            },
                            backends={
                                "depth": depth_result.backend,
                                "filter": filter_result.backend,
                                "semantic": semantic_result.backend,
                                "decision": "local_rules",
                            },
                            error="",
                        )
                    else:
                        assert qwen_client is not None
                        self.state.update_status(
                            status="qwen_thinking",
                            frame_index=frame_index,
                            decision_text="Qwen3-VL 正在分析当前帧。",
                            backends={"depth": "-", "filter": "-", "semantic": "-", "decision": "qwen3_vl"},
                            error="",
                        )
                        cv2.imwrite(str(qwen_frame_path), frame)
                        prompt = build_qwen_ui_prompt(frame_index)
                        decision_text = qwen_client.ask_image(
                            qwen_frame_path,
                            prompt,
                            max_new_tokens=int(min(180, int(vlm_cfg["qwen3_vl_max_new_tokens"]))),
                        )
                        decision_text = sanitize_qwen_decision_text(decision_text)
                        risk_level = extract_risk_level(decision_text)
                        nearest = parse_qwen_nearest(decision_text)
                        self.state.update_status(
                            status="running",
                            frame_index=frame_index,
                            decision_text=decision_text,
                            risk_level=risk_level,
                            nearest=nearest,
                            detections=[],
                            timings={"total_ms": round(now_ms() - loop_start, 2)},
                            backends={"depth": "-", "filter": "-", "semantic": "-", "decision": "qwen3_vl"},
                            error="",
                        )
                else:
                    self.state.update_status(status="running", frame_index=frame_index)

                frame_index += 1
                if self.frame_delay_ms > 0:
                    time.sleep(self.frame_delay_ms / 1000.0)

            final_status = "stopped" if self.stop_event.is_set() else "finished"
            self.state.update_status(status=final_status)
        except Exception as exc:
            self.state.update_status(status="error", error=str(exc))
        finally:
            if cap is not None:
                cap.release()


class DecisionHTTPRequestHandler(BaseHTTPRequestHandler):
    server: "DecisionHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(HTML_PAGE)
        elif parsed.path == "/api/videos":
            self.send_json(self.handle_videos())
        elif parsed.path == "/api/status":
            self.send_json(self.server.state.snapshot_status())
        elif parsed.path == "/stream":
            self.handle_stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            try:
                self.send_json(self.handle_start())
            except (RuntimeError, ValueError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        elif parsed.path == "/api/stop":
            self.server.state.stop()
            self.send_json({"ok": True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def handle_videos(self) -> dict[str, Any]:
        videos = list_video_files(resolve_path("data/input"))
        configured = resolve_path(self.server.state.config["paths"]["input_video"])
        if configured.exists() and configured not in videos:
            videos.insert(0, configured)
        return {
            "videos": [{"name": path.name, "path": relative_label(path)} for path in videos],
            "defaults": {
                "max_frames": self.server.state.default_max_frames,
                "decision_stride": self.server.state.default_decision_stride,
                "decision_backend": self.server.state.default_decision_backend,
                "frame_delay_ms": self.server.state.default_frame_delay_ms,
                "backends": [
                    {"value": "qwen3_vl", "label": "Qwen3-VL"},
                    {"value": "local_rules", "label": "本地规则"},
                ],
            },
        }

    def handle_start(self) -> dict[str, Any]:
        payload = self.read_json()
        video_arg = str(payload.get("video") or "")
        video_path = resolve_path(video_arg)
        if not video_path.exists() or video_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"视频不存在或格式不支持：{video_arg}")
        max_frames = int(payload.get("max_frames") or 0)
        decision_stride = int(payload.get("decision_stride") or self.server.state.default_decision_stride)
        decision_backend = str(payload.get("decision_backend") or self.server.state.default_decision_backend)
        if decision_backend not in {"qwen3_vl", "local_rules"}:
            raise ValueError(f"不支持的决策后端：{decision_backend}")
        frame_delay_ms = int(payload.get("frame_delay_ms") or self.server.state.default_frame_delay_ms)
        self.server.state.start(video_path, max_frames, decision_stride, decision_backend, frame_delay_ms)
        return {"ok": True, "video": relative_label(video_path)}

    def handle_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        last_version = -1
        while True:
            frame, version = self.server.state.snapshot_frame()
            if version != last_version:
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last_version = version
                except (BrokenPipeError, ConnectionResetError):
                    break
            time.sleep(0.06)

    def send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            raise ValueError("请求体不是有效 JSON")


class DecisionHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: DecisionUIState) -> None:
        super().__init__(server_address, DecisionHTTPRequestHandler)
        self.state = state


def list_video_files(input_dir: Path) -> list[Path]:
    input_dir = resolve_path(input_dir)
    if not input_dir.exists():
        return []
    videos = [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    return sorted(videos, key=lambda item: str(item).lower())


def relative_label(path: Path) -> str:
    root = resolve_path(".")
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def extract_risk_level(decision_text: str) -> str:
    match = re.search(r"风险等级：([^。\n]+)", decision_text)
    return match.group(1).strip() if match else "-"


def parse_qwen_nearest(decision_text: str) -> dict[str, Any] | None:
    target_match = re.search(r"最近目标：([^。\n]+)", decision_text)
    z_match = re.search(r"(?:伪距离\s*Z|距离判断|距离)：([^。\n]+)", decision_text)
    if not target_match and not z_match:
        return None
    return {
        "class_name": target_match.group(1).strip() if target_match else "见 Qwen3-VL 判断",
        "confidence": None,
        "xyxy": [0, 0, 0, 0],
        "median_z": None,
        "z_text": z_match.group(1).strip() if z_match else "视觉估计",
        "center_overlap": False,
    }


def sanitize_qwen_decision_text(decision_text: str) -> str:
    """避免 Qwen3-VL 在纯 RGB 模式下把视觉估计写成真实米制距离。"""

    text = re.sub(r"距离约\s*\d+(?:\.\d+)?\s*(?:米|m)", "视觉估计距离", decision_text, flags=re.IGNORECASE)
    text = re.sub(r"约\s*\d+(?:\.\d+)?\s*(?:米|m)", "视觉估计", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:米|m)", "视觉估计距离", text, flags=re.IGNORECASE)
    return text.strip()


def build_qwen_ui_prompt(frame_index: int) -> str:
    return (
        "你是移动机器人/车辆的视觉语言决策模块。请只根据当前 RGB 原始视频帧进行实时避障决策，"
        "不要编造真实米制深度，禁止输出任何数字米制距离，例如“2 米”“2.5m”“约 3 米”。\n"
        f"当前帧序号：{frame_index}。\n"
        "请严格按下面 4 行中文格式输出：\n"
        "风险等级：低/中/高。\n"
        "最近目标：写出最需要关注的目标、所在方位。\n"
        "距离判断：只能写近/中等/远或无法判断；说明这是视觉估计，不是真实米制距离。\n"
        "建议：给出一句可执行动作。"
    )


def placeholder_jpeg(text: str) -> bytes:
    image = np.full((540, 960, 3), (22, 25, 19), dtype=np.uint8)
    cv2.putText(image, text, (360, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 236, 224), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return b""
    return encoded.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动实时决策 UI")
    parser.add_argument("--config", default="configs/demo.yaml", help="配置文件路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--max-frames", type=int, default=None, help="默认最多处理帧数；0 表示处理到视频结束")
    parser.add_argument("--decision-backend", choices=["qwen3_vl", "local_rules"], default="qwen3_vl", help="默认决策后端")
    parser.add_argument("--decision-stride", type=int, default=None, help="每隔多少帧更新一次决策")
    parser.add_argument("--frame-delay-ms", type=int, default=250, help="每帧播放延迟，Qwen3-VL 模式建议适当放慢")
    args = parser.parse_args()

    config = load_config(args.config)
    default_max_frames = int(args.max_frames if args.max_frames is not None else config["runtime"]["max_frames"])
    if args.decision_stride is None:
        default_decision_stride = 30 if args.decision_backend == "qwen3_vl" else 1
    else:
        default_decision_stride = max(1, int(args.decision_stride))
    state = DecisionUIState(
        config,
        default_max_frames,
        default_decision_stride,
        args.decision_backend,
        max(0, int(args.frame_delay_ms)),
    )
    server = DecisionHTTPServer((args.host, args.port), state)
    url = f"http://{args.host}:{args.port}"
    print(f"实时决策 UI 已启动：{url}")
    print("打开页面后选择视频并点击启动。按 Ctrl+C 结束服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop()
        server.server_close()


if __name__ == "__main__":
    main()
