from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GeneratorConfig, ZingConfig
from .action import ActionConditioner
from .attention import CompiledSegment, CrossAttention, SelfAttention, compute_rope, make_rope_freqs
from .kv_cache import CausalKVCache


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("time embedding dimension must be even")
    half = dim // 2
    values = position.to(torch.float64)
    frequencies = torch.pow(10000, -torch.arange(half, device=values.device).to(values).div(half))
    sinusoid = torch.outer(values, frequencies)
    return torch.cat((torch.cos(sinusoid), torch.sin(sinusoid)), dim=1)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool = False, compile_fusion: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.compile_fusion = compile_fusion

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return CompiledSegment.get(self._norm, self.compile_fusion and value.is_cuda)(
            value, self.weight, self.bias, self.eps
        )

    @staticmethod
    def _norm(
        value: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float
    ) -> torch.Tensor:
        return F.layer_norm(
            value.float(),
            (value.shape[-1],),
            None if weight is None else weight.float(),
            None if bias is None else bias.float(),
            eps,
        ).type_as(value)


def adaln_op(
    x,
    m_shift=None,
    m_scale=None,
    e_shift=None,
    e_scale=None,
    eps=1.0e-6,
    y=None,
    m_gate=None,
    e_gate=None,
    r=None,
    weight=None,
    bias=None,
    cast_norm=False,
):
    if y is not None:
        x = (x.float() + y.float() * (m_gate.float() + e_gate.float())).type_as(x)
    if r is not None:
        x = x + r
    if m_shift is None:
        return x
    h = F.layer_norm(
        x.float(),
        (x.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    )
    shift, scale = m_shift + e_shift, m_scale + e_scale
    if cast_norm:
        h = h.type_as(x)
        return x, h * (1 + scale) + shift
    return x, (h * (1 + scale.float()) + shift.float()).type_as(x)


def adaln(value: torch.Tensor, compile_fusion: bool, *args, **kwargs):
    function = CompiledSegment.get(adaln_op, compile_fusion and value.is_cuda)
    return function(value, *args, **kwargs)


class WanAttentionBlock(nn.Module):
    def __init__(self, config: GeneratorConfig, deterministic: bool):
        super().__init__()
        self.compile_fusion = config.compile_fusion
        self.norm1 = WanLayerNorm(config.dim, config.eps, compile_fusion=config.compile_fusion)
        self.self_attn = SelfAttention(
            config.dim,
            config.num_heads,
            config.eps,
            config.qk_norm,
            config.compile_fusion,
            deterministic,
        )
        self.norm3 = (
            WanLayerNorm(
                config.dim,
                config.eps,
                elementwise_affine=True,
                compile_fusion=config.compile_fusion,
            )
            if config.cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = CrossAttention(
            config.dim,
            config.num_heads,
            config.eps,
            config.qk_norm,
            config.compile_fusion,
            deterministic,
        )
        self.norm2 = WanLayerNorm(config.dim, config.eps, compile_fusion=config.compile_fusion)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, config.dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim**0.5)

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        self_history: tuple[torch.Tensor | None, torch.Tensor | None],
        cross_history: tuple[torch.Tensor | None, torch.Tensor | None],
        cache=None,
        layer: int | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        _, normalized = adaln(
            hidden,
            self.compile_fusion,
            self.modulation[:, 0],
            self.modulation[:, 1],
            time_embedding.select(-2, 0),
            time_embedding.select(-2, 1),
            self.norm1.eps,
        )
        attended, key, value = self.self_attn(
            normalized, rope[0], rope[1], self_history, cache=cache, layer=layer
        )
        hidden = adaln(
            hidden,
            self.compile_fusion,
            y=attended,
            m_gate=self.modulation[:, 2],
            e_gate=time_embedding.select(-2, 2),
        )
        crossed, new_cross = self.cross_attn(self.norm3(hidden), context, context_lengths, cross_history)
        hidden, normalized = adaln(
            hidden,
            self.compile_fusion,
            self.modulation[:, 3],
            self.modulation[:, 4],
            time_embedding.select(-2, 3),
            time_embedding.select(-2, 4),
            self.norm2.eps,
            r=crossed,
        )
        feed_forward = self.ffn(normalized)
        hidden = adaln(
            hidden,
            self.compile_fusion,
            y=feed_forward,
            m_gate=self.modulation[:, 5],
            e_gate=time_embedding.select(-2, 5),
        )
        return hidden, (key, value), new_cross


class WanHead(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.compile_fusion = config.compile_fusion
        self.norm = WanLayerNorm(config.dim, config.eps, compile_fusion=config.compile_fusion)
        self.head = nn.Linear(config.dim, math.prod(config.patch_size) * config.out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim**0.5)

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        _, normalized = adaln(
            hidden,
            self.compile_fusion,
            self.modulation[:, 0],
            self.modulation[:, 1],
            embedding,
            embedding,
            self.norm.eps,
        )
        return self.head(normalized)


class WanModel(nn.Module):
    def __init__(self, config: ZingConfig):
        super().__init__()
        generator = config.generator
        self.config = generator
        self.patch_size = tuple(generator.patch_size)
        self.out_dim = generator.out_dim
        self.rope_max_seq_len = generator.rope_max_seq_len
        self.frames_per_block = config.inference.frames_per_block
        self.patch_embedding = nn.Conv3d(
            generator.in_dim, generator.dim, self.patch_size, stride=self.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(generator.text_dim, generator.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(generator.dim, generator.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(generator.freq_dim, generator.dim),
            nn.SiLU(),
            nn.Linear(generator.dim, generator.dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(generator.dim, generator.dim * 6))
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(generator, config.inference.deterministic_attention) for _ in range(generator.num_layers)]
        )
        self.head = WanHead(generator)
        self.freqs_t = None
        self.freqs_h = None
        self.freqs_w = None
        self._init_weights()
        self.action_in = ActionConditioner(config.action, generator.dim)
        self.action_history_frames = 2 * (config.action.kernel_size - 1)

    def _init_weights(self) -> None:
        if self.patch_embedding.weight.is_meta:
            return
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.head.head.weight)

    def make_kv_cache(self) -> CausalKVCache:
        return CausalKVCache(
            len(self.blocks),
            self.action_history_frames,
            local_attn_size=self.config.local_attn_size,
            sink_size=self.config.sink_size,
            frames_per_block=self.frames_per_block,
            rope_max_seq_len=self.rope_max_seq_len,
        )

    def _rope(self, positions: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.freqs_t is None:
            self.freqs_t, self.freqs_h, self.freqs_w = make_rope_freqs(
                self.config.dim, self.config.num_heads, self.rope_max_seq_len
            )
        if self.freqs_t.device != reference.device:
            self.freqs_t = self.freqs_t.to(reference.device)
            self.freqs_h = self.freqs_h.to(reference.device)
            self.freqs_w = self.freqs_w.to(reference.device)
        return compute_rope(positions, self.freqs_t, self.freqs_h, self.freqs_w)

    def _pack_embed(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        grids = []
        for sample in inputs:
            embedded = self.patch_embedding(sample.unsqueeze(0))
            grids.append(torch.tensor(embedded.shape[2:], dtype=torch.long, device=inputs.device))
            tokens.append(embedded.flatten(2).transpose(1, 2).squeeze(0))
        return torch.cat(tokens, dim=0), torch.stack(grids)

    def _unpatchify(self, tokens: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        channels = self.out_dim
        shape = [int(value) for value in grid.tolist()]
        value = tokens.view(*shape, *self.patch_size, channels)
        value = torch.einsum("fhwpqrc->cfphqwr", value)
        return value.reshape(channels, *[a * b for a, b in zip(shape, self.patch_size)])

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        cache: CausalKVCache,
        cache_mode: str | None,
        action: torch.Tensor | None = None,
        prompt_switch: bool = False,
    ) -> torch.Tensor:
        packed, grid_sizes = self._pack_embed(inputs)
        batch = inputs.shape[0]
        sequence = packed.shape[0] // batch
        hidden = packed.view(batch, sequence, -1)
        positions, block_id, action_window = cache.prepare(grid_sizes, inputs.device, action, prompt_switch)
        if action_window is not None:
            hidden = self.action_in.add_action_to_tokens(hidden, action_window, grid_sizes, align_to_end=True)
        frames, height, width = (int(value) for value in grid_sizes[0].tolist())
        spatial_tokens = height * width
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(batch, frames)
        timestep = timestep[:, :frames]
        frame_index = torch.arange(frames, device=inputs.device).repeat_interleave(spatial_tokens)
        embedded_time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.freq_dim, timestep.reshape(-1)).to(hidden.dtype)
        )
        time_projection = self.time_projection(embedded_time).view(batch, frames, 6, -1)[:, frame_index]
        head_embedding = embedded_time.view(batch, frames, -1)[:, frame_index]
        query_positions, key_positions = cache.attention_positions(positions)
        rope = self._rope(query_positions, hidden), self._rope(key_positions, hidden)
        embedded_context = self.text_embedding(context)
        new_self = []
        new_cross = []
        for index, block in enumerate(self.blocks):
            hidden, self_values, cross_values = block(
                hidden,
                time_projection,
                rope,
                embedded_context,
                context_lengths,
                cache.history(index),
                cache.cross(index),
                cache,
                index,
            )
            new_self.append(self_values)
            new_cross.append(cross_values)
        cache.update(new_self, new_cross, positions, block_id, cache_mode, action)
        output = self.head(hidden, head_embedding)
        return torch.stack([self._unpatchify(output[index], grid_sizes[index]) for index in range(batch)])
