# gfx1151 / ROCm adaptation notes

This is a fork-local draft for running Zing-0.5 on AMD Ryzen AI MAX+ 395
(Strix Halo, `gfx1151`). It is not an official Seedleap platform.

## Hard gaps versus the NVIDIA release

| Upstream pin | gfx1151 status |
| --- | --- |
| Linux + NVIDIA CUDA | Host kernel/driver must already be ROCm 10.0.0 (`amdrocm-core-sdk10.0-gfx1151`) |
| `torch==2.9.1` from PyPI | CUDA wheel. Install a gfx1151 HIP wheel instead |
| `flash-attn==2.8.3.post1` | CUDA extension. This fork falls back to PyTorch SDPA |
| `97/9` sliding window, ≥80 GB | First bring-up uses `33/5` plus a 5-frame smoke JSONL |
| Real-time 24 FPS | Out of scope for the first pass. Goal is a finite video |

`torch.cuda.is_available()` and `cuda:0` remain the ROCm device API. That is
not the blocker.

## Attention fallback

`src/zing_v0_5/model/attention.py` keeps the packed varlen call shape used by
self-attention and cross-attention:

1. `ZING_ATTENTION_BACKEND=auto` (default): use `flash_attn` when it imports,
   otherwise SDPA.
2. `sdpa`: PyTorch `scaled_dot_product_attention` (AOTriton/efficient/math as
   selected by the wheel).
3. `sdpa-math`: force the math backend for a correctness A/B.
4. `flash`: require CUDA `flash-attn`; fail loudly if missing.

Equal-length batches reshape packed `(total, heads, dim)` back to
`(batch, heads, seq, dim)`. Unequal prompt lengths (cross-attention with
`batch > 1`) run one SDPA per sample and concatenate. Official generate()
encodes one prompt at a time, so the equal-length path is the hot path.

Do not install PyPI `flash-attn` into a ROCm venv. A CUDA extension that
"installs successfully" is not a HIP attention backend.

## Python dependency list

Install PyTorch **before** `requirements-rocm.txt`, and verify HIP plus the
exact architecture before any other package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# Preferred on a ROCm 10.0.0 host: TheRock multi-arch index, gfx1151 extra,
# pinned to a 10.0.0 tag. Do not float to the latest 10.1.0 nightly.
python -m pip install --pre \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  --extra-index-url https://pypi.org/simple \
  "torch[device-gfx1151]==2.14.0a0+rocm10.0.0a20260730" \
  torchvision torchaudio

python - <<'PY'
import torch
print(torch.__version__, torch.version.hip)
assert torch.cuda.is_available() and torch.version.hip
print(torch.cuda.get_device_properties(0).gcnArchName)
x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
print((x @ x).sum().item())
PY

python -m pip install -r requirements-rocm.txt
```

If the `device-gfx1151` extra is missing on that tag, install the same
`torch==2.14.0a0+rocm10.0.0a20260730` pin without the extra and abort unless
`gcnArchName` reports `gfx1151`.

Do **not** use:

- PyPI `torch==2.9.1` or `flash-attn==2.8.3.post1`
- `https://repo.amd.com/rocm/whl/gfx1151/` as the first choice on a ROCm 10
  host: that index currently stops at ROCm 7.13 wheels
- `https://rocm.nightlies.amd.com/v2/gfx1151/` as the first choice: same 7.x
  ceiling as of 2026-09-06

`requirements-rocm.txt` matches upstream for everything except torch and
flash-attn. `torchvision` / `torchaudio` must come from the same AMD index as
`torch`, not from PyPI.

## UMA memory on lab-apu-max395

Measured split: ~96 GiB GPU-addressable UMA, ~30 GiB system RAM. Weights are
about 34 GB (`generator/model.pt` ~20 GB fp32, UMT5 ~11 GB, VAE ~2.8 GB).

- Keep weights in `~/.cache` or another durable path. A reboot wipes `/tmp`.
- `pipeline.py` mmap-loads the generator on CPU, then `.to("cuda:0")`, while
  UMT5 and VAE stay on CPU until a generate/decode step. Do not materialize
  the 20 GB state dict and the 11 GB encoder in the 30 GiB system partition
  at the same time.
- First run: `--local-attn-size 33 --sink-size 5` and
  `examples/rocm_gfx1151_smoke.jsonl` (5 frames, 480x832). Do not start with
  official 241-frame 704x1280 JSONL or `97/9`.
- Leave `compile_fusion: false`. Do not enable `torch.compile` on the first pass.

## Bring-up command

```bash
ZING_ATTENTION_BACKEND=sdpa-math \
CUDA_VISIBLE_DEVICES=0 \
HIP_VISIBLE_DEVICES=0 \
ZING_PYTHON=/path/to/.venv/bin/python \
bash run.sh \
  --pretrained-dir /path/to/Zing-0.5/pretrained \
  --checkpoint /path/to/Zing-0.5/generator/model.pt \
  --messages examples/rocm_gfx1151_smoke.jsonl \
  --output-dir outputs/gfx1151-smoke \
  --local-attn-size 33 \
  --sink-size 5 \
  --seed 0
```

If `sdpa-math` produces a finite video, retry with `ZING_ATTENTION_BACKEND=sdpa`
to pick up AOTriton/efficient kernels. Black or NaN frames are an attention or
bf16-kernel issue until proven otherwise; do not treat a single-board result as
an architecture-wide claim.

## Still not done in this draft

- Generator load order / streaming H2D to shrink the 30 GiB system-RAM peak
- AITER or ROCm FlashAttention as a later kernel, after SDPA works
- Real-time 24 FPS, `97/9`, or an upstream Seedleap PR
