from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import load_config, with_cache_window, with_compile_fusion
from .io import iter_message_jsonl, output_stem, write_media
from .pipeline import InferencePipeline
from .processor import MessageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="zing-0.5")
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-attn-size", dest="local_attn_size", type=int)
    parser.add_argument("--sink-size", dest="sink_size", type=int)
    parser.add_argument(
        "--config.generator_config.local_attn_size",
        dest="local_attn_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config.generator_config.sink_size",
        dest="sink_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--compile-fusion", action="store_true")
    parser.add_argument("--bench-json", dest="bench_json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config_path = Path(__file__).resolve().parents[2] / "config" / "zing.yaml"
    config = load_config(config_path)
    config = with_cache_window(config, args.local_attn_size, args.sink_size)
    config = with_compile_fusion(config, True if args.compile_fusion else None)
    torch.set_grad_enabled(False)
    pipeline = InferencePipeline(config, args.pretrained_dir, args.checkpoint)
    processor = MessageProcessor(config, pipeline.encode_reference)
    set_seed(args.seed)
    benches = []
    with torch.no_grad():
        for line_index, sample in iter_message_jsonl(args.messages):
            request = processor.process(sample)
            video = pipeline.generate(request)
            frames = video[0].permute(0, 2, 3, 1).mul(255).to(torch.uint8)
            target = write_media(args.output_dir, output_stem(sample, line_index), frames, config.inference.output_fps)
            print(target, flush=True)
            if pipeline.last_bench is not None:
                record = dict(pipeline.last_bench)
                record["output"] = str(target)
                record["sample_id"] = sample.get("sample_id")
                benches.append(record)
                print(json.dumps({"bench": record}, default=str), flush=True)
    if args.bench_json:
        Path(args.bench_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.bench_json).write_text(json.dumps(benches, indent=2) + "\n", encoding="utf-8")
