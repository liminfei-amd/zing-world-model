#!/usr/bin/env python3
"""Local browser demo: WASD/IJKL + mid-rollout prompt. Not the Loopit product UI."""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

os.environ.setdefault("ZING_KEEP_TEXT_ENCODER", "1")
os.environ.setdefault("ZING_STREAM_VAE", "1")
os.environ.setdefault("ZING_KEEP_VAE", "1")
os.environ.setdefault("ZING_SESSION_OFFLOAD_TEXT", "1")
os.environ.setdefault("ZING_ATTENTION_BACKEND", "sdpa-flash")
os.environ.setdefault("ZING_COMPILE", "generator")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
import sys

sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from zing_v0_5.config import load_config, with_cache_window
from zing_v0_5.pipeline import InferencePipeline
from zing_v0_5.session import WorldSession, vram_snapshot

KEY_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Zing-0.5 本地交互试用</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 15px/1.45 system-ui, sans-serif; background: #111; color: #eee; }
  main { max-width: 960px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 8px; }
  .note { color: #aaa; margin-bottom: 16px; }
  img#view { width: 100%; background: #000; border: 1px solid #333; display: block; }
  .row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
  input[type=text] { flex: 1; min-width: 220px; padding: 8px 10px; background: #1b1b1b; color: #eee; border: 1px solid #444; }
  button { padding: 8px 14px; background: #2a2a2a; color: #eee; border: 1px solid #555; cursor: pointer; }
  button:hover { background: #333; }
  kbd { display: inline-block; min-width: 1.4em; padding: 2px 6px; margin: 0 2px; border: 1px solid #555; background: #1a1a1a; }
  kbd.on { background: #3d5a3d; border-color: #6a8; }
  #stats { margin-top: 10px; color: #9ad; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<main>
  <h1>Zing-0.5 本地交互试用（R9700）</h1>
  <p class="note">这不是 Loopit 官方 App，也不是 24 FPS。33/5 滑窗会丢掉窗口外的 KV，世界可以一直往前滚，直到你点停止。编码后会把 UMT5 卸到 CPU，避免 32 GiB 卡被常驻权重撑满。点「开始世界」后点画面，再用 W/A/S/D 移动、I/J/K/L 视角。</p>
  <img id="view" alt="world" width="832" height="480"/>
  <div class="row">
    <input id="prompt" type="text" value="A quiet lake at sunrise, first-person view, gentle camera motion"/>
    <button id="start">开始世界</button>
    <button id="rewrite">改写世界</button>
    <button id="stop">停止</button>
  </div>
  <p>移动 <kbd id="k-w">W</kbd><kbd id="k-a">A</kbd><kbd id="k-s">S</kbd><kbd id="k-d">D</kbd>
     视角 <kbd id="k-i">I</kbd><kbd id="k-j">J</kbd><kbd id="k-k">K</kbd><kbd id="k-l">L</kbd></p>
  <p id="stats">未开始</p>
</main>
<script>
const keys = {w:0,a:0,s:0,d:0,i:0,j:0,k:0,l:0};
const view = document.getElementById('view');
const stats = document.getElementById('stats');
function paintKeys() {
  for (const k of Object.keys(keys)) {
    document.getElementById('k-'+k).classList.toggle('on', keys[k] === 1);
  }
}
function send(path, body) {
  return fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body||{})});
}
document.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k in keys) { keys[k] = 1; paintKeys(); e.preventDefault(); send('/keys', keys); }
});
document.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  if (k in keys) { keys[k] = 0; paintKeys(); e.preventDefault(); send('/keys', keys); }
});
let starting = false;
document.getElementById('start').onclick = async () => {
  if (starting) return;
  starting = true;
  stats.textContent = '正在切换会话…';
  try {
    await send('/stop', {});
    const r = await send('/start', {prompt: document.getElementById('prompt').value});
    const body = await r.json();
    if (!r.ok || body.ok === false) {
      stats.textContent = body.error || '无法开始（上一轮还在停）';
      return;
    }
    view.src = '/stream?' + Date.now();
  } finally {
    starting = false;
  }
};
document.getElementById('rewrite').onclick = () => send('/prompt', {prompt: document.getElementById('prompt').value});
document.getElementById('stop').onclick = () => send('/stop', {});
async function poll() {
  try {
    const r = await fetch('/status');
    const s = await r.json();
    stats.textContent = s.text;
  } catch (e) {}
  setTimeout(poll, 400);
}
poll();
</script>
</body>
</html>
"""


class DemoState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lifecycle = threading.Lock()
        self.keys = {name: 0.0 for name in KEY_ORDER}
        self.pending_prompt: str | None = None
        self.jpeg = b""
        self.frame_event = threading.Event()
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.status = "模型加载中…"
        self.session: WorldSession | None = None


STATE = DemoState()


def _jpeg(frame) -> bytes:
    image = Image.fromarray(frame.numpy())
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _keys_tensor(session: WorldSession):
    import torch

    with STATE.lock:
        values = [STATE.keys[name] for name in KEY_ORDER]
    return torch.tensor(values, dtype=torch.float32, device=session.device)


def generate_loop(prompt: str) -> None:
    session = STATE.session
    assert session is not None
    try:
        STATE.status = "正在创建世界…"
        session.start(prompt, 480, 832)
        if STATE.stop.is_set():
            STATE.status = "已停止"
            return
        STATE.status = "生成中（无限，点停止结束）"
        started = time.perf_counter()
        pixels_done = 0
        while not STATE.stop.is_set():
            with STATE.lock:
                rewrite = STATE.pending_prompt
                STATE.pending_prompt = None
            if rewrite:
                session.set_prompt(rewrite)
                STATE.status = "已改写 prompt，继续生成"
            block = session.step(_keys_tensor(session))
            if block is None:
                STATE.status = "会话已结束"
                break
            for frame in block:
                STATE.jpeg = _jpeg(frame)
                STATE.frame_event.set()
                pixels_done += 1
            elapsed = time.perf_counter() - started
            fps = pixels_done / elapsed if elapsed > 0 else 0.0
            snap = vram_snapshot()
            kv = 0 if session.cache is None else session.cache.used_tokens
            cap = 0 if session.cache is None or session.cache.kv_capacity is None else session.cache.kv_capacity
            STATE.status = (
                f"已出 {pixels_done} 帧 · 会话约 {fps:.2f} FPS · 墙钟 {elapsed:.1f}s · "
                f"显存 {snap['allocated_gib']:.1f}/{snap['reserved_gib']:.1f}G 空闲 {snap['free_gib']:.1f}G · "
                f"KV {kv}/{cap}"
            )
    except Exception as exc:
        STATE.status = f"出错：{exc}"
        raise
    finally:
        session.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/status":
            payload = json.dumps({"text": STATE.status}).encode("utf-8")
            self._send(200, payload, "application/json")
            return
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    STATE.frame_event.wait(timeout=1.0)
                    STATE.frame_event.clear()
                    jpeg = STATE.jpeg
                    if not jpeg:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                    self.wfile.write(str(len(jpeg)).encode("ascii"))
                    self.wfile.write(b"\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except BrokenPipeError:
                return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}
        if self.path == "/keys":
            with STATE.lock:
                for name in KEY_ORDER:
                    if name in data:
                        STATE.keys[name] = 1.0 if data[name] else 0.0
            self._send(200, b'{"ok":true}', "application/json")
            return
        if self.path == "/prompt":
            with STATE.lock:
                STATE.pending_prompt = str(data.get("prompt") or "")
            self._send(200, b'{"ok":true}', "application/json")
            return
        if self.path == "/stop":
            STATE.stop.set()
            self._send(200, b'{"ok":true}', "application/json")
            return
        if self.path == "/start":
            prompt = str(data.get("prompt") or "A quiet lake at sunrise")
            with STATE.lifecycle:
                STATE.stop.set()
                if STATE.worker is not None:
                    STATE.worker.join(timeout=60)
                if STATE.worker is not None and STATE.worker.is_alive():
                    self._send(
                        409,
                        json.dumps({"ok": False, "error": "上一轮还在停，请稍后再点开始"}).encode("utf-8"),
                        "application/json",
                    )
                    return
                STATE.stop.clear()
                STATE.worker = threading.Thread(target=generate_loop, args=(prompt,), daemon=True)
                STATE.worker.start()
            self._send(200, b'{"ok":true}', "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--pretrained-dir", default=str(Path.home() / ".cache/zing-0.5/pretrained"))
    parser.add_argument("--checkpoint", default=str(Path.home() / ".cache/zing-0.5/generator/model.pt"))
    args = parser.parse_args()
    config_path = ROOT / "config" / "zing.yaml"
    config = with_cache_window(load_config(config_path), 33, 5)
    print("loading pipeline…", flush=True)
    pipeline = InferencePipeline(config, args.pretrained_dir, args.checkpoint)
    STATE.session = WorldSession(pipeline, config)
    STATE.status = "就绪：打开页面后点「开始世界」"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"READY http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
