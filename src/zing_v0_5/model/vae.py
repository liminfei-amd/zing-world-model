from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify


def _env_flag_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _patch_residual_up_skip_clone() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualUpBlock

    if getattr(WanResidualUpBlock.forward, "_zing_skip_clone", False):
        return

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        skip = x
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx) if feat_cache is not None else resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache, feat_idx) if feat_cache is not None else self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(skip, first_chunk=first_chunk)
        return x

    forward._zing_skip_clone = True
    WanResidualUpBlock.forward = forward


class WanVAE(torch.nn.Module):
    def __init__(self, pretrained_dir: str | Path):
        super().__init__()
        self.model = AutoencoderKLWan.from_pretrained(Path(pretrained_dir) / "vae")
        self.model.eval().requires_grad_(False)
        self.model.clear_cache()
        self.mean = torch.tensor(self.model.config.latents_mean, dtype=torch.float32)
        self.std = torch.tensor(self.model.config.latents_std, dtype=torch.float32)
        self._stream_index = 0
        self._runtime_ready = False

    def prepare_runtime(self) -> None:
        if self._runtime_ready:
            return
        self._runtime_ready = True
        if torch.cuda.is_available() and _env_flag_default("ZING_CUDNN_BENCHMARK", False):
            torch.backends.cudnn.benchmark = True
        if _env_flag_default("ZING_VAE_SKIP_UP_CLONE", True):
            _patch_residual_up_skip_clone()

    def _sdpa_context(self):
        # Wan mid-block attention is a single head with dim=1024. ROCm FLASH
        # rejects last-dim > 256 and mem-efficient rejects > 512, then aborts
        # if those backends are forced. Auto therefore pins MATH.
        name = os.environ.get("ZING_VAE_SDPA", "auto").strip().lower() or "auto"
        if name in {"auto", "off", "none"}:
            name = "math"
        if name == "eager":
            return nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mapping = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
        }
        if name not in mapping:
            raise ValueError(f"unsupported ZING_VAE_SDPA={name!r}")
        return sdpa_kernel(mapping[name])

    def _stats(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(device=reference.device, dtype=reference.dtype).view(1, -1, 1, 1, 1)
        std = self.std.to(device=reference.device, dtype=reference.dtype).view(1, -1, 1, 1, 1)
        return mean, std

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.uint8:
            raise ValueError("reference frames must use uint8 pixels")
        dtype = next(self.model.parameters()).dtype
        pixels = frames.to(device=next(self.model.parameters()).device, dtype=dtype).div(127.5).sub(1.0)
        mean, std = self._stats(pixels)
        self.model.clear_cache()
        latent = (self.model.encode(pixels).latent_dist.mode() - mean) / std
        self.model.clear_cache()
        return latent.float().permute(0, 2, 1, 3, 4)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        latents = latents.to(device=next(self.model.parameters()).device, dtype=dtype)
        mean, std = self._stats(latents)
        value = latents.permute(0, 2, 1, 3, 4) * std + mean
        self.model.clear_cache()
        decoded = self.model.decode(value).sample.float()
        self.model.clear_cache()
        return decoded.permute(0, 2, 1, 3, 4)

    def begin_stream(self) -> None:
        if getattr(self.model, "use_tiling", False):
            raise RuntimeError("stream VAE does not support tiled AutoencoderKLWan decode")
        self.prepare_runtime()
        self.model.clear_cache()
        self._stream_index = 0

    def decode_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the next latent frames, keeping causal feat_cache across calls."""
        if latents.ndim != 5:
            raise ValueError("decode_chunk expects BTHWC-style latents with 5 dims")
        dtype = next(self.model.parameters()).dtype
        latents = latents.to(device=next(self.model.parameters()).device, dtype=dtype)
        mean, std = self._stats(latents)
        value = latents.permute(0, 2, 1, 3, 4) * std + mean
        hidden = self.model.post_quant_conv(value)
        frames = []
        with self._sdpa_context():
            for index in range(hidden.shape[2]):
                self.model._conv_idx = [0]
                first_chunk = self._stream_index == 0
                frame = self.model.decoder(
                    hidden[:, :, index : index + 1, :, :],
                    feat_cache=self.model._feat_map,
                    feat_idx=self.model._conv_idx,
                    first_chunk=first_chunk,
                )
                frames.append(frame)
                self._stream_index += 1
        decoded = torch.cat(frames, dim=2)
        patch_size = self.model.config.patch_size
        if patch_size is not None:
            decoded = unpatchify(decoded, patch_size=patch_size)
        decoded = torch.clamp(decoded, min=-1.0, max=1.0)
        return decoded.float().permute(0, 2, 1, 3, 4)

    def end_stream(self) -> None:
        self.model.clear_cache()
        self._stream_index = 0
