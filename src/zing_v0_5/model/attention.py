from __future__ import annotations

import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


os.environ.setdefault("TRITON_MAX_BLOCK_X", "8192")
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._inductor.config.realize_opcount_threshold = 100
torch._dynamo.config.recompile_limit = 1024


class CompiledSegment:
    compiled = {}
    mode = None

    @classmethod
    def get(cls, function, enabled: bool):
        if not enabled:
            return function
        if function not in cls.compiled:
            options = {}
            if "recompile_limit" in inspect.signature(torch.compile).parameters:
                options["recompile_limit"] = 1024
            cls.compiled[function] = torch.compile(function, dynamic=True, mode=cls.mode, **options)
        return cls.compiled[function]


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, compile_fusion: bool):
        super().__init__()
        self.eps = eps
        self.compile_fusion = compile_fusion
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return CompiledSegment.get(self._norm, self.compile_fusion and value.is_cuda)(value, self.weight, self.eps)

    @staticmethod
    def _norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        value_float = value.float()
        return (value_float * torch.rsqrt(value_float.pow(2).mean(dim=-1, keepdim=True) + eps)).type_as(value) * weight


def make_rope_freqs(dim: int, num_heads: int, maximum: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def table(length: int, width: int) -> torch.Tensor:
        positions = torch.arange(length)
        frequencies = 1.0 / torch.pow(10000, torch.arange(0, width, 2).to(torch.float64).div(width))
        angles = torch.outer(positions, frequencies)
        return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).float()

    head_dim = dim // num_heads
    temporal_width = head_dim - 4 * (head_dim // 6)
    spatial_width = 2 * (head_dim // 6)
    return table(maximum, temporal_width), table(maximum, spatial_width), table(maximum, spatial_width)


def _is_compiling() -> bool:
    compiling = getattr(torch.compiler, "is_compiling", None)
    return compiling is not None and compiling()


def compute_rope(
    positions: torch.Tensor, temporal: torch.Tensor, height: torch.Tensor, width: torch.Tensor
) -> torch.Tensor:
    if not _is_compiling():
        maxima = positions.max(dim=0).values
        if int(maxima[0]) >= temporal.shape[0] or int(maxima[1]) >= height.shape[0] or int(maxima[2]) >= width.shape[0]:
            raise ValueError("RoPE position exceeds generator.rope_max_seq_len")
    return torch.cat((temporal[positions[:, 0]], height[positions[:, 1]], width[positions[:, 2]]), dim=1)


def apply_rope(value: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    sequence = value.shape[-3]
    head_dim = value.shape[-1]
    shaped = rope.reshape(*([1] * (value.dim() - 3)), sequence, 1, head_dim // 2, 2)
    cosine, sine = shaped[..., 0], shaped[..., 1]
    real, imaginary = value[..., 0::2].float(), value[..., 1::2].float()
    rotated = torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), dim=-1)
    return rotated.flatten(-2).to(value.dtype)


_ATTENTION_BACKEND_CACHE: tuple[str, str] | None = None


def _requested_attention_backend() -> str:
    return os.environ.get("ZING_ATTENTION_BACKEND", "auto").strip().lower() or "auto"


def _flash_attn_varlen_func():
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func


def _resolve_attention_backend() -> str:
    global _ATTENTION_BACKEND_CACHE
    requested = _requested_attention_backend()
    if _ATTENTION_BACKEND_CACHE is not None and _ATTENTION_BACKEND_CACHE[0] == requested:
        return _ATTENTION_BACKEND_CACHE[1]
    allowed = {"auto", "flash", "sdpa", "sdpa-math", "sdpa-flash", "sdpa-efficient"}
    if requested not in allowed:
        raise ValueError(f"unsupported ZING_ATTENTION_BACKEND={requested!r}; expected one of {sorted(allowed)}")
    flash_available = True
    try:
        _flash_attn_varlen_func()
    except ImportError:
        flash_available = False
    if requested == "flash":
        if not flash_available:
            raise ImportError("ZING_ATTENTION_BACKEND=flash requires the CUDA flash-attn package")
        resolved = "flash"
    elif requested in {"sdpa", "sdpa-math", "sdpa-flash", "sdpa-efficient"}:
        resolved = requested
    else:
        resolved = "flash" if flash_available else "sdpa"
    _ATTENTION_BACKEND_CACHE = (requested, resolved)
    return resolved


def _sdpa_backend_name(resolved: str) -> str:
    if resolved == "sdpa-math":
        return "math"
    if resolved == "sdpa-flash":
        return "flash"
    if resolved == "sdpa-efficient":
        return "efficient"
    return "auto"


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    backend: str,
) -> torch.Tensor:
    kwargs = {"dropout_p": 0.0, "is_causal": False}
    if backend == "auto":
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    mapping = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    if backend not in mapping:
        raise ValueError(f"unsupported SDPA kernel {backend!r}")
    with sdpa_kernel(mapping[backend]):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


def _sdpa_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    *,
    backend: str,
) -> torch.Tensor:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("packed attention tensors must have shape (total, heads, dim)")
    if query.shape[1:] != key.shape[1:] or key.shape[1:] != value.shape[1:]:
        raise ValueError("query, key, and value head shapes must match")
    batch = query_lengths.shape[0]
    heads, dim = query.shape[1], query.shape[2]
    if _is_compiling():
        max_query = query.shape[0] // batch
        max_key = key.shape[0] // batch
        packed_query = query.view(batch, max_query, heads, dim).transpose(1, 2)
        packed_key = key.view(batch, max_key, heads, dim).transpose(1, 2)
        packed_value = value.view(batch, max_key, heads, dim).transpose(1, 2)
        attended = _scaled_dot_product_attention(packed_query, packed_key, packed_value, backend=backend)
        return attended.transpose(1, 2).reshape(query.shape)
    query_lengths = query_lengths.to(device=query.device, dtype=torch.int64)
    key_lengths = key_lengths.to(device=key.device, dtype=torch.int64)
    if query_lengths.ndim != 1 or key_lengths.ndim != 1 or query_lengths.numel() != key_lengths.numel():
        raise ValueError("query_lengths and key_lengths must be 1-D and the same batch size")
    if int(query_lengths.sum().item()) != query.shape[0] or int(key_lengths.sum().item()) != key.shape[0]:
        raise ValueError("packed token counts must match the length tensors")
    batch = int(query_lengths.numel())
    max_query = int(query_lengths.max().item())
    max_key = int(key_lengths.max().item())
    equal_query = bool((query_lengths == max_query).all().item())
    equal_key = bool((key_lengths == max_key).all().item())
    if equal_query and equal_key:
        packed_query = query.view(batch, max_query, heads, dim).transpose(1, 2)
        packed_key = key.view(batch, max_key, heads, dim).transpose(1, 2)
        packed_value = value.view(batch, max_key, heads, dim).transpose(1, 2)
        attended = _scaled_dot_product_attention(packed_query, packed_key, packed_value, backend=backend)
        return attended.transpose(1, 2).reshape(query.shape)

    outputs = []
    query_offset = 0
    key_offset = 0
    for index in range(batch):
        query_len = int(query_lengths[index].item())
        key_len = int(key_lengths[index].item())
        sample_query = query[query_offset : query_offset + query_len].unsqueeze(0).transpose(1, 2)
        sample_key = key[key_offset : key_offset + key_len].unsqueeze(0).transpose(1, 2)
        sample_value = value[key_offset : key_offset + key_len].unsqueeze(0).transpose(1, 2)
        attended = _scaled_dot_product_attention(sample_query, sample_key, sample_value, backend=backend)
        outputs.append(attended.transpose(1, 2).squeeze(0))
        query_offset += query_len
        key_offset += key_len
    return torch.cat(outputs, dim=0)


def _flash_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    flash_attn_varlen_func = _flash_attn_varlen_func()
    cumulative_query = F.pad(query_lengths.to(torch.int32).cumsum(0), (1, 0)).to(torch.int32)
    cumulative_key = F.pad(key_lengths.to(torch.int32).cumsum(0), (1, 0)).to(torch.int32)
    return flash_attn_varlen_func(
        query,
        key,
        value,
        cumulative_query,
        cumulative_key,
        int(query_lengths.max().item()),
        int(key_lengths.max().item()),
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        deterministic=deterministic,
    )


def _stage_kv_enabled() -> bool:
    return os.environ.get("ZING_STAGE_KV", "1").strip().lower() not in {"0", "false", "no", "off"}


def flash_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    backend = _resolve_attention_backend()
    if backend == "flash":
        return _flash_attention_varlen(query, key, value, query_lengths, key_lengths, deterministic)
    return _sdpa_attention_varlen(
        query,
        key,
        value,
        query_lengths,
        key_lengths,
        backend="math" if deterministic else _sdpa_backend_name(backend),
    )


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, qk_norm: bool, compile_fusion: bool, deterministic: bool):
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by the head count")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.deterministic = deterministic
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        history: tuple[torch.Tensor | None, torch.Tensor | None],
        cache=None,
        layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        query = self.q(hidden)
        key = self.k(hidden)
        if isinstance(self.norm_q, WanRMSNorm):
            query = WanRMSNorm._norm(query, self.norm_q.weight, self.norm_q.eps)
            key = WanRMSNorm._norm(key, self.norm_k.weight, self.norm_k.eps)
        query = query.reshape(batch, sequence, self.num_heads, self.head_dim)
        key = key.reshape(batch, sequence, self.num_heads, self.head_dim)
        value = self.v(hidden).reshape(batch, sequence, self.num_heads, self.head_dim)
        staged = None
        if cache is not None and layer is not None and _stage_kv_enabled():
            staged = cache.stage_current(layer, key, value)
        if staged is not None:
            raw_key, full_value = staged
        else:
            history_key, history_value = history
            raw_key = key if history_key is None else torch.cat((history_key, key), dim=1)
            full_value = value if history_value is None else torch.cat((history_value, value), dim=1)
        query = apply_rope(query, query_rope)
        rotated_key = apply_rope(raw_key, key_rope)
        key_sequence = rotated_key.shape[1]
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        key_lengths = torch.full((batch,), key_sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query.reshape(batch * sequence, self.num_heads, self.head_dim),
            rotated_key.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            full_value.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            query_lengths,
            key_lengths,
            self.deterministic,
        )
        attended = attended.reshape(batch, sequence, self.num_heads * self.head_dim)
        return self.o(attended), key, value


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, qk_norm: bool, compile_fusion: bool, deterministic: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.deterministic = deterministic
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        cached: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, sequence, _ = hidden.shape
        query = self.norm_q(self.q(hidden)).reshape(batch * sequence, self.num_heads, self.head_dim)
        key, value = cached
        created = None
        if key is None:
            key = self.norm_k(self.k(context)).reshape(context.shape[0], self.num_heads, self.head_dim)
            value = self.v(context).reshape(context.shape[0], self.num_heads, self.head_dim)
            created = (key, value)
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query, key, value, query_lengths, context_lengths.to(torch.int32), self.deterministic
        )
        return self.o(attended.reshape(batch, sequence, -1)), created
