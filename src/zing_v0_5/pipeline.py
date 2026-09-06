from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from .config import ZingConfig
from .model import WanModel, WanTextEncoder, WanVAE
from .processor import InferenceRequest
from .scheduler import DmdScheduler


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class InferencePipeline:
    def __init__(self, config: ZingConfig, pretrained_dir: str | Path, checkpoint: str | Path):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA or ROCm HIP GPU is required")
        self.config = config
        self.device = torch.device("cuda:0")
        self.skip_cache_final = _env_flag("ZING_SKIP_CACHE_FINAL")
        self.decode_per_block = _env_flag("ZING_DECODE_PER_BLOCK")
        self.last_bench: dict | None = None
        pretrained_path = Path(pretrained_dir)
        for name in ("text_encoder", "tokenizer", "vae"):
            if not (pretrained_path / name).is_dir():
                raise ValueError(f"pretrained directory is missing {name}/")
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.suffix != ".pt" or not checkpoint_path.is_file():
            raise ValueError("checkpoint must be an existing .pt file")
        self.generator = self._load_generator(config, checkpoint_path)
        compile_mode = os.environ.get("ZING_COMPILE", "").strip().lower()
        if compile_mode in {"generator", "max-autotune"}:
            mode = "max-autotune" if compile_mode == "max-autotune" else "default"
            self.generator = torch.compile(self.generator, dynamic=True, mode=mode)
        self.text_encoder = WanTextEncoder(pretrained_path, config.text_encoder.max_length)
        self.text_encoder.eval().requires_grad_(False).to(device="cpu", dtype=torch.bfloat16)
        self.vae = WanVAE(pretrained_path)
        self.vae.eval().requires_grad_(False).to(device="cpu", dtype=torch.bfloat16)

    def _unwrap_generator(self):
        return getattr(self.generator, "_orig_mod", self.generator)

    def _load_generator(self, config: ZingConfig, checkpoint_path: Path) -> WanModel:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, dict) or not state or not all(isinstance(key, str) for key in state):
            raise ValueError("checkpoint must contain a bare state dict")
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("checkpoint state dict values must all be tensors")
        with torch.device("meta"):
            generator = WanModel(config)
        expected = set(generator.state_dict())
        actual = set(state)
        if expected != actual:
            raise ValueError(
                f"checkpoint keys mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        generator = generator.to_empty(device=self.device).to(dtype=torch.bfloat16)
        with torch.no_grad():
            for name, dest in generator.state_dict().items():
                source = state[name]
                dest.copy_(source.to(device=dest.device, dtype=dest.dtype))
        del state
        return generator.eval().requires_grad_(False)

    def encode_reference(self, frames: torch.Tensor) -> torch.Tensor:
        self.vae.to(self.device)
        try:
            return self.vae.encode(frames).cpu()
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

    @staticmethod
    def _known(latents: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        known = (~mask).view(mask.shape[0], mask.shape[1], 1, 1, 1)
        return torch.where(known, clean, latents)

    def _model_flow(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: tuple[torch.Tensor, torch.Tensor],
        cache,
        cache_mode: str | None,
        action: torch.Tensor | None,
        prompt_switch: bool,
    ) -> torch.Tensor:
        value = latents.permute(0, 2, 1, 3, 4)
        output = self.generator(
            value,
            timestep,
            context[0],
            context[1],
            cache,
            cache_mode,
            action=action,
            prompt_switch=prompt_switch,
        )
        return output.permute(0, 2, 1, 3, 4)

    def _timed_flow(self, *args, **kwargs) -> tuple[torch.Tensor, float]:
        _synchronize()
        started = time.perf_counter()
        output = self._model_flow(*args, **kwargs)
        _synchronize()
        return output, time.perf_counter() - started

    def generate(self, request: InferenceRequest) -> torch.Tensor:
        wall_start = time.perf_counter()
        _synchronize()
        encode_start = time.perf_counter()
        self.text_encoder.to(self.device)
        contexts = [self.text_encoder.encode([prompt]) for prompt in request.prompts]
        self.text_encoder.to("cpu")
        _synchronize()
        encode_s = time.perf_counter() - encode_start
        clean = request.clean_latents.to(device=self.device, dtype=torch.bfloat16)
        mask = request.label_mask.to(device=self.device, dtype=torch.bool)
        action = None if request.action is None else request.action.to(self.device)
        latents = self._known(torch.randn_like(clean), clean, mask)
        output = clean.clone()
        lengths = [int(value) for value in request.prompt_lengths.tolist()]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cache = self._unwrap_generator().make_kv_cache()
        segment = 0
        last_end = request.chunk_spans[-1][1]
        dit_s = 0.0
        forwards = 0
        first_block_s = None
        pixel_scale = self.config.vae.temporal_scale
        for start, end in request.chunk_spans:
            previous_segment = segment
            while segment + 1 < len(contexts) and start >= boundaries[segment + 1]:
                segment += 1
            prompt_switch = segment != previous_segment
            context = (
                contexts[segment][0].to(device=self.device, dtype=torch.bfloat16),
                contexts[segment][1].to(device=self.device),
            )
            current = latents[:, start:end]
            current_clean = clean[:, start:end]
            current_mask = mask[:, start:end]
            current_action = None if action is None else action[:, start:end]
            keep_cache = end < last_end
            if keep_cache:
                patch = self.config.generator.patch_size
                token_count = (
                    (end - start) // patch[0]
                    * (current.shape[3] // patch[1])
                    * (current.shape[4] // patch[2])
                )
                cache.reserve(token_count)
            if not bool(current_mask.any()):
                if keep_cache:
                    zero = torch.zeros((current.shape[0], end - start), device=self.device, dtype=torch.float32)
                    _, elapsed = self._timed_flow(
                        current_clean, zero, context, cache, "final", current_action, prompt_switch
                    )
                    dit_s += elapsed
                    forwards += 1
                output[:, start:end] = current_clean
                continue
            scheduler = DmdScheduler(self.config.inference).to(self.device)
            cache_mode = "active" if keep_cache else None
            block_start = time.perf_counter()
            for step_index, timestep in enumerate(scheduler.timesteps):
                step_switch = prompt_switch and step_index == 0
                step_time = timestep * torch.ones(
                    (current.shape[0], end - start), device=self.device, dtype=torch.float32
                )
                step_time = torch.where(current_mask, step_time, 0)
                current = self._known(current, current_clean, current_mask)
                prediction, elapsed = self._timed_flow(
                    current, step_time, context, cache, cache_mode, current_action, step_switch
                )
                dit_s += elapsed
                forwards += 1
                current, _ = scheduler.step(prediction, current)
                current = self._known(current, current_clean, current_mask)
            output[:, start:end] = current
            if keep_cache:
                if self.skip_cache_final:
                    cache.commit_active(current_action)
                else:
                    zero = torch.zeros((current.shape[0], end - start), device=self.device, dtype=torch.float32)
                    _, elapsed = self._timed_flow(current, zero, context, cache, "final", current_action, False)
                    dit_s += elapsed
                    forwards += 1
            _synchronize()
            block_s = time.perf_counter() - block_start
            if first_block_s is None and bool(current_mask.any()):
                first_block_s = block_s
                if self.decode_per_block:
                    self.vae.to(self.device)
                    try:
                        _ = self.vae.decode(output[:, :end])
                    finally:
                        self.vae.to("cpu")
        del contexts, cache, latents, clean, mask, action
        torch.cuda.empty_cache()
        _synchronize()
        vae_start = time.perf_counter()
        self.vae.to(self.device)
        try:
            video = self.vae.decode(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            _synchronize()
            vae_s = time.perf_counter() - vae_start
            cpu_video = video.cpu()
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()
        wall_s = time.perf_counter() - wall_start
        pixel_frames = int(cpu_video.shape[1])
        finite = bool(torch.isfinite(cpu_video).all().item())
        frame_mean = float(cpu_video.mean().item()) if finite else float("nan")
        black_frac = float((cpu_video < 0.02).all(dim=2).float().mean().item()) if finite else 1.0
        self.last_bench = {
            "pixel_frames": pixel_frames,
            "latent_frames": int(output.shape[1]),
            "pixel_scale": pixel_scale,
            "forwards": forwards,
            "encode_s": encode_s,
            "dit_s": dit_s,
            "vae_s": vae_s,
            "wall_s": wall_s,
            "first_block_s": first_block_s,
            "dit_fps": (pixel_frames / dit_s) if dit_s > 0 else 0.0,
            "e2e_fps": (pixel_frames / wall_s) if wall_s > 0 else 0.0,
            "skip_cache_final": self.skip_cache_final,
            "decode_per_block": self.decode_per_block,
            "finite": finite,
            "frame_mean": frame_mean,
            "black_frac": black_frac,
        }
        return cpu_video
