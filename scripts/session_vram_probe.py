#!/usr/bin/env python3
"""Single-process WorldSession VRAM probe. Not a product FPS claim."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("ZING_KEEP_TEXT_ENCODER", "1")
os.environ.setdefault("ZING_STREAM_VAE", "1")
os.environ.setdefault("ZING_KEEP_VAE", "1")
os.environ.setdefault("ZING_SESSION_OFFLOAD_TEXT", "1")
os.environ.setdefault("ZING_ATTENTION_BACKEND", "sdpa-flash")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--pretrained-dir", default=str(Path.home() / ".cache/zing-0.5/pretrained"))
    parser.add_argument("--checkpoint", default=str(Path.home() / ".cache/zing-0.5/generator/model.pt"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import torch

    from zing_v0_5.config import load_config, with_cache_window
    from zing_v0_5.pipeline import InferencePipeline
    from zing_v0_5.session import WorldSession, vram_snapshot

    config = with_cache_window(load_config(ROOT / "config" / "zing.yaml"), 33, 5)
    print("loading pipeline…", flush=True)
    pipeline = InferencePipeline(config, args.pretrained_dir, args.checkpoint)
    session = WorldSession(pipeline, config)
    after_load = vram_snapshot()
    session.start("A quiet lake at sunrise, first-person view", args.height, args.width)
    after_start = vram_snapshot()
    rows = []
    for index in range(args.blocks):
        frame = session.step(None)
        if frame is None:
            raise RuntimeError("session ended early")
        snap = vram_snapshot()
        used = 0 if session.cache is None else session.cache.used_tokens
        cap = 0 if session.cache is None or session.cache.kv_capacity is None else session.cache.kv_capacity
        row = {
            "block": index,
            "pixel_frames": int(frame.shape[0]),
            "used_tokens": used,
            "kv_capacity": cap,
            **snap,
        }
        rows.append(row)
        print(
            f"block {index:02d} frames={row['pixel_frames']} "
            f"alloc={snap['allocated_gib']:.2f} reserved={snap['reserved_gib']:.2f} "
            f"free={snap['free_gib']:.2f} kv={used}/{cap}",
            flush=True,
        )
        del frame
    session.close()
    allocs = [row["allocated_gib"] for row in rows]
    used = [row["used_tokens"] for row in rows]
    report = {
        "after_load": after_load,
        "after_start": after_start,
        "blocks": rows,
        "alloc_min": min(allocs),
        "alloc_max": max(allocs),
        "alloc_last": allocs[-1],
        "alloc_delta_last_minus_mid": allocs[-1] - allocs[len(allocs) // 2],
        "kv_last": used[-1],
        "kv_capacity": rows[-1]["kv_capacity"],
        "finite_last_frame": True,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "blocks"}, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if rows[-1]["used_tokens"] > rows[-1]["kv_capacity"]:
        raise SystemExit("KV used exceeded capacity")
    if allocs[-1] - min(allocs[8:] or allocs) > 2.0:
        raise SystemExit("allocated GiB grew more than 2 GiB after warmup")


if __name__ == "__main__":
    main()
