from __future__ import annotations

import os
import time
from contextlib import nullcontext

import torch

from .config import ZingConfig
from .pipeline import InferencePipeline
from .scheduler import DmdScheduler


def _env_flag_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def vram_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0, "free_gib": 0.0, "total_gib": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
    }


class WorldSession:
    """Unbounded causal rollout. KV stays in the 33/5 window; only the current block is allocated."""

    def __init__(self, pipeline: InferencePipeline, config: ZingConfig):
        if not pipeline.stream_vae:
            raise RuntimeError("WorldSession requires ZING_STREAM_VAE=1")
        self.pipeline = pipeline
        self.config = config
        self.device = pipeline.device
        self.offload_text_encoder = _env_flag_default("ZING_SESSION_OFFLOAD_TEXT", True)
        self.pipeline_vae = _env_flag_default("ZING_PIPELINE_VAE", False)
        self._reset_state()

    def _reset_state(self) -> None:
        self.cache = None
        self.context = None
        self.block_index = 0
        self.stream_started = False
        self.prompt_switch = False
        self.latent_h = 0
        self.latent_w = 0
        self.pixel_h = 0
        self.pixel_w = 0
        self.alive = False
        self._scheduler = None
        self._dit_stream = None
        self._vae_stream = None
        self._pending_event = None
        self._pending_video = None
        self.last_dit_ms = 0.0
        self.last_vae_ms = 0.0

    def close(self) -> None:
        self._consume_pending()
        if self.stream_started:
            try:
                self.pipeline.vae.end_stream()
            except Exception:
                pass
        self._reset_state()
        torch.cuda.empty_cache()

    def _text_encoder_on_device(self) -> bool:
        return next(self.pipeline.text_encoder.parameters()).device == self.device

    def _encode_prompt(self, prompt: str) -> None:
        encoder = self.pipeline.text_encoder
        if not self._text_encoder_on_device():
            encoder.to(self.device)
        try:
            encoded = encoder.encode([prompt])
            self.context = (
                encoded[0].to(device=self.device, dtype=torch.bfloat16),
                encoded[1].to(self.device),
            )
        finally:
            if self.offload_text_encoder:
                encoder.to("cpu")
                torch.cuda.empty_cache()

    def start(self, prompt: str, height: int, width: int, max_pixel_frames: int | None = None) -> None:
        del max_pixel_frames
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        self.close()
        self.latent_h = height // self.config.vae.spatial_scale
        self.latent_w = width // self.config.vae.spatial_scale
        self.pixel_h = height
        self.pixel_w = width
        self.pipeline._ensure_vae_device()
        self._encode_prompt(prompt)
        generator = self.pipeline._unwrap_generator()
        self.cache = generator.make_kv_cache()
        patch = generator.patch_size
        tokens_per_frame = (self.latent_h // patch[1]) * (self.latent_w // patch[2])
        self.cache.allocate_window(
            tokens_per_frame,
            generator.config.num_heads,
            generator.config.dim // generator.config.num_heads,
            self.device,
            torch.bfloat16,
        )
        self.block_index = 0
        self.stream_started = False
        self.prompt_switch = False
        self.alive = True
        self._scheduler = DmdScheduler(self.config.inference).to(self.device)
        if self.pipeline_vae:
            self._dit_stream = torch.cuda.Stream()
            self._vae_stream = torch.cuda.Stream()

    def set_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt or not self.alive:
            return
        self._encode_prompt(prompt)
        self.prompt_switch = True

    def _action_chunk(self, latent_frames: int, keys: torch.Tensor | None) -> torch.Tensor | None:
        if keys is None:
            return None
        factor = self.config.vae.temporal_scale
        chunk = torch.zeros((1, latent_frames, factor, 8), device=self.device, dtype=torch.float32)
        if self.block_index > 0:
            chunk[...] = keys.view(1, 1, 1, 8)
        return chunk

    def _trim_action_history(self) -> None:
        history = self.cache.action_history
        keep = self.cache.action_history_frames
        if history is None or keep <= 0 or history.shape[1] <= keep:
            return
        self.cache.action_history = history[:, -keep:].contiguous()

    def _reclaim_if_tight(self) -> None:
        snap = vram_snapshot()
        if snap["free_gib"] < 1.0 or snap["reserved_gib"] > 0.90 * snap["total_gib"]:
            torch.cuda.empty_cache()

    def _video_to_cpu(self, video: torch.Tensor) -> torch.Tensor:
        return video[0].permute(0, 2, 3, 1).mul(255).to(torch.uint8).cpu()

    def _consume_pending(self) -> torch.Tensor | None:
        if self._pending_event is None:
            return None
        self._pending_event.synchronize()
        video = self._pending_video
        self._pending_event = None
        self._pending_video = None
        cpu = self._video_to_cpu(video)
        del video
        return cpu

    def _launch_vae(self, latents: torch.Tensor) -> None:
        pipeline = self.pipeline
        stream = self._vae_stream
        chunk = latents.contiguous()
        if stream is None:
            if not self.stream_started:
                pipeline.vae.begin_stream()
                self.stream_started = True
            pixels = pipeline.vae.decode_chunk(chunk)
            video = (pixels * 0.5 + 0.5).clamp(0, 1)
            self._pending_event = torch.cuda.Event()
            self._pending_event.record()
            self._pending_video = video
            return
        stream.wait_stream(torch.cuda.current_stream() if self._dit_stream is None else self._dit_stream)
        with torch.cuda.stream(stream):
            if not self.stream_started:
                pipeline.vae.begin_stream()
                self.stream_started = True
            pixels = pipeline.vae.decode_chunk(chunk)
            video = (pixels * 0.5 + 0.5).clamp(0, 1)
            event = torch.cuda.Event()
            event.record()
        self._pending_event = event
        self._pending_video = video

    def step(self, keys: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.alive:
            return None
        pipeline = self.pipeline
        latent_frames = 1 if self.block_index == 0 else self.config.inference.frames_per_block
        z_dim = self.config.vae.z_dim
        profile = os.environ.get("ZING_SESSION_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
        if profile:
            torch.cuda.synchronize()
            dit_start = time.perf_counter()
        infer = torch.inference_mode()
        with infer:
            return self._step_body(keys, latent_frames, z_dim, profile, dit_start if profile else 0.0)

    def _step_body(
        self,
        keys: torch.Tensor | None,
        latent_frames: int,
        z_dim: int,
        profile: bool,
        dit_start: float,
    ) -> torch.Tensor | None:
        pipeline = self.pipeline
        dit_stream = self._dit_stream
        if dit_stream is not None:
            dit_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(dit_stream) if dit_stream is not None else nullcontext():
            current = torch.randn(
                (1, latent_frames, z_dim, self.latent_h, self.latent_w),
                device=self.device,
                dtype=torch.bfloat16,
            )
            current_clean = torch.zeros_like(current)
            current_mask = torch.ones((1, latent_frames), device=self.device, dtype=torch.bool)
            current_action = self._action_chunk(latent_frames, keys)
            prompt_switch = self.prompt_switch
            self.prompt_switch = False
            scheduler = self._scheduler
            scheduler.reset()
            step_sync = os.environ.get("ZING_SESSION_STEP_SYNC", "").strip().lower() in {"1", "true", "yes", "on"}
            flow = pipeline._timed_flow if step_sync else None
            for step_index, timestep in enumerate(scheduler.timesteps):
                step_switch = prompt_switch and step_index == 0
                step_time = timestep * torch.ones(
                    (current.shape[0], latent_frames), device=self.device, dtype=torch.float32
                )
                if flow is None:
                    prediction = pipeline._model_flow(
                        current, step_time, self.context, self.cache, "active", current_action, step_switch
                    )
                else:
                    prediction, _ = flow(
                        current, step_time, self.context, self.cache, "active", current_action, step_switch
                    )
                current, _ = scheduler.step(prediction, current)
            if pipeline.skip_cache_final:
                self.cache.commit_active(current_action)
            else:
                zero = torch.zeros((current.shape[0], latent_frames), device=self.device, dtype=torch.float32)
                if flow is None:
                    pipeline._model_flow(current, zero, self.context, self.cache, "final", current_action, False)
                else:
                    flow(current, zero, self.context, self.cache, "final", current_action, False)
            self._trim_action_history()
        if profile:
            if dit_stream is not None:
                dit_stream.synchronize()
            else:
                torch.cuda.synchronize()
            dit_end = time.perf_counter()
        if self.pipeline_vae:
            previous = self._consume_pending()
            self._launch_vae(current)
            if previous is None:
                cpu = torch.zeros((0, self.pixel_h, self.pixel_w, 3), dtype=torch.uint8)
            else:
                cpu = previous
        else:
            self._launch_vae(current)
            cpu = self._consume_pending()
        self.block_index += 1
        del current, current_clean, current_mask, current_action
        if profile:
            torch.cuda.synchronize()
            self.last_dit_ms = (dit_end - dit_start) * 1000.0
            self.last_vae_ms = (time.perf_counter() - dit_end) * 1000.0
        self._reclaim_if_tight()
        return cpu
