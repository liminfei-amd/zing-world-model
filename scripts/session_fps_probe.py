#!/usr/bin/env python3
"""Warm WorldSession FPS. Discard warmup blocks; not a 1-generate-per-process claim."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("ZING_KEEP_TEXT_ENCODER", "1")
os.environ.setdefault("ZING_STREAM_VAE", "1")
os.environ.setdefault("ZING_KEEP_VAE", "1")
os.environ.setdefault("ZING_SESSION_OFFLOAD_TEXT", "1")
os.environ.setdefault("ZING_PIPELINE_VAE", "0")
os.environ.setdefault("ZING_ATTENTION_BACKEND", "sdpa-flash")
os.environ.setdefault("ZING_COMPILE", "generator")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--pretrained-dir", default=str(Path.home() / ".cache/zing-0.5/pretrained"))
    parser.add_argument("--checkpoint", default=str(Path.home() / ".cache/zing-0.5/generator/model.pt"))
    parser.add_argument("--out", default="")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.profile:
        os.environ["ZING_SESSION_PROFILE"] = "1"

    import torch

    from zing_v0_5.config import load_config, with_cache_window
    from zing_v0_5.pipeline import InferencePipeline
    from zing_v0_5.session import WorldSession, vram_snapshot

    config = with_cache_window(load_config(ROOT / "config" / "zing.yaml"), 33, 5)
    print("loading pipeline…", flush=True)
    pipeline = InferencePipeline(config, args.pretrained_dir, args.checkpoint)
    session = WorldSession(pipeline, config)
    session.start("A quiet lake at sunrise, first-person view", args.height, args.width)
    pixels = 0
    for index in range(args.warmup):
        frame = session.step(None)
        if frame is None:
            raise RuntimeError("session ended during warmup")
        pixels += int(frame.shape[0])
        print(f"warmup {index:02d} frames={int(frame.shape[0])}", flush=True)
        del frame
    torch.cuda.synchronize()
    started = time.perf_counter()
    timed_pixels = 0
    last_mean = float("nan")
    dit_ms = []
    vae_ms = []
    for index in range(args.blocks):
        frame = session.step(None)
        if frame is None:
            raise RuntimeError("session ended during timed blocks")
        timed_pixels += int(frame.shape[0])
        last_mean = float(frame.float().mean().item())
        if args.profile:
            dit_ms.append(session.last_dit_ms)
            vae_ms.append(session.last_vae_ms)
        del frame
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    snap = vram_snapshot()
    used = 0 if session.cache is None else session.cache.used_tokens
    cap = 0 if session.cache is None or session.cache.kv_capacity is None else session.cache.kv_capacity
    session.close()
    report = {
        "warmup_blocks": args.warmup,
        "timed_blocks": args.blocks,
        "timed_pixels": timed_pixels,
        "elapsed_s": elapsed,
        "e2e_fps": timed_pixels / elapsed if elapsed > 0 else 0.0,
        "ms_per_block": (elapsed / args.blocks) * 1000.0 if args.blocks else 0.0,
        "last_frame_mean": last_mean,
        "kv_used": used,
        "kv_capacity": cap,
        "stage_kv": os.environ.get("ZING_STAGE_KV", "1"),
        "step_sync": os.environ.get("ZING_SESSION_STEP_SYNC", "0"),
        "attention": os.environ.get("ZING_ATTENTION_BACKEND", "sdpa-flash"),
        "compile": os.environ.get("ZING_COMPILE", ""),
        "pipeline_vae": os.environ.get("ZING_PIPELINE_VAE", "0"),
        "vae_sdpa": os.environ.get("ZING_VAE_SDPA", "auto"),
        "compile_fusion": os.environ.get("ZING_COMPILE_FUSION", "0"),
        "vae_skip_up_clone": os.environ.get("ZING_VAE_SKIP_UP_CLONE", "1"),
        **snap,
    }
    if dit_ms:
        report["dit_ms_mean"] = sum(dit_ms) / len(dit_ms)
        report["vae_ms_mean"] = sum(vae_ms) / len(vae_ms)
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if last_mean != last_mean:
        raise SystemExit("last frame mean is NaN")


if __name__ == "__main__":
    main()
