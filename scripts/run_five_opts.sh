#!/usr/bin/env bash
# Five sequential gfx1151 realtime optimizations. One process per experiment.
set -euo pipefail

ZING_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ZING_PYTHON=${ZING_PYTHON:-python3}
PRETRAINED=${ZING_PRETRAINED_DIR:?set ZING_PRETRAINED_DIR}
CHECKPOINT=${ZING_CHECKPOINT:?set ZING_CHECKPOINT}
OUT_ROOT=${ZING_OUT_ROOT:-"$ZING_ROOT/outputs/gfx1151-five-opts"}
MESSAGES=${ZING_MESSAGES:-"$ZING_ROOT/examples/rocm_gfx1151_bench.jsonl"}
export PYTHONPATH="$ZING_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:False}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ATTN_FAST="${ZING_OPT_ATTN:-sdpa}"
TUNABLE_FILE="$OUT_ROOT/tunableop.csv"

run_opt() {
  local name="$1"
  shift
  local dest="$OUT_ROOT/$name"
  mkdir -p "$dest"
  echo "===== $name ====="
  local extra=()
  if [[ "${COMPILE_FUSION:-0}" == 1 ]]; then
    extra+=(--compile-fusion)
  fi
  set +e
  env "$@" "$ZING_PYTHON" -m zing_v0_5 \
    --pretrained-dir "$PRETRAINED" \
    --checkpoint "$CHECKPOINT" \
    --messages "$MESSAGES" \
    --output-dir "$dest" \
    --local-attn-size 33 \
    --sink-size 5 \
    --seed 0 \
    --bench-json "$dest/bench.json" \
    "${extra[@]}" | tee "$dest/stdout.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$rc" >"$dest/exit_code.txt"
  return "$rc"
}

mkdir -p "$OUT_ROOT"

echo "===== opt1 MATH SDPA ====="
run_opt opt1-sdpa-math ZING_ATTENTION_BACKEND=sdpa-math

echo "===== opt2 fused SDPA ====="
if run_opt opt2-sdpa-flash ZING_ATTENTION_BACKEND=sdpa-flash; then
  ATTN_FAST=sdpa-flash
else
  echo "FLASH SDPA failed; trying EFFICIENT"
  if run_opt opt2-sdpa-efficient ZING_ATTENTION_BACKEND=sdpa-efficient; then
    ATTN_FAST=sdpa-efficient
  else
    echo "fused SDPA failed; keeping MATH for later opts"
    ATTN_FAST=sdpa-math
  fi
fi
printf '%s\n' "$ATTN_FAST" >"$OUT_ROOT/attn_fast.txt"

echo "===== opt3 TunableOp tune + replay ====="
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="$TUNABLE_FILE"
export PYTORCH_TUNABLEOP_TUNING=1
run_opt opt3-tunableop-tune ZING_ATTENTION_BACKEND="$ATTN_FAST"
export PYTORCH_TUNABLEOP_TUNING=0
run_opt opt3-tunableop-replay ZING_ATTENTION_BACKEND="$ATTN_FAST"

echo "===== opt4 compile_fusion + TunableOp replay ====="
COMPILE_FUSION=1
run_opt opt4-compile-fusion ZING_ATTENTION_BACKEND="$ATTN_FAST"

echo "===== opt5 skip cache-final + per-block decode ====="
run_opt opt5-skip-final-block-decode \
  ZING_ATTENTION_BACKEND="$ATTN_FAST" \
  ZING_SKIP_CACHE_FINAL=1 \
  ZING_DECODE_PER_BLOCK=1

echo "Wrote $OUT_ROOT"
find "$OUT_ROOT" -name bench.json -print
