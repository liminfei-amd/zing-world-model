from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ["ZING_ATTENTION_BACKEND"] = "sdpa-math"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from zing_v0_5.model.attention import compute_rope, flash_attention_varlen, make_rope_freqs


def _naive_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    outputs = []
    query_offset = 0
    key_offset = 0
    for query_len, key_len in zip(query_lengths.tolist(), key_lengths.tolist()):
        sample_query = query[query_offset : query_offset + query_len].transpose(0, 1).float()
        sample_key = key[key_offset : key_offset + key_len].transpose(0, 1).float()
        sample_value = value[key_offset : key_offset + key_len].transpose(0, 1).float()
        scores = torch.matmul(sample_query, sample_key.transpose(-1, -2)) * scale
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(weights, sample_value).transpose(0, 1).to(query.dtype))
        query_offset += int(query_len)
        key_offset += int(key_len)
    return torch.cat(outputs, dim=0)


class SdpaVarlenFallbackTests(unittest.TestCase):
    def test_equal_lengths_match_naive_attention(self) -> None:
        torch.manual_seed(0)
        batch, query_len, key_len, heads, dim = 2, 4, 6, 3, 8
        query = torch.randn(batch * query_len, heads, dim)
        key = torch.randn(batch * key_len, heads, dim)
        value = torch.randn(batch * key_len, heads, dim)
        query_lengths = torch.full((batch,), query_len, dtype=torch.int32)
        key_lengths = torch.full((batch,), key_len, dtype=torch.int32)
        actual = flash_attention_varlen(query, key, value, query_lengths, key_lengths, True)
        expected = _naive_varlen(query, key, value, query_lengths, key_lengths)
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=1e-5, rtol=1e-4))

    def test_unequal_key_lengths_match_naive_attention(self) -> None:
        torch.manual_seed(1)
        heads, dim = 2, 8
        query_lengths = torch.tensor([3, 5], dtype=torch.int32)
        key_lengths = torch.tensor([4, 2], dtype=torch.int32)
        query = torch.randn(int(query_lengths.sum()), heads, dim)
        key = torch.randn(int(key_lengths.sum()), heads, dim)
        value = torch.randn(int(key_lengths.sum()), heads, dim)
        actual = flash_attention_varlen(query, key, value, query_lengths, key_lengths, True)
        expected = _naive_varlen(query, key, value, query_lengths, key_lengths)
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=1e-5, rtol=1e-4))

    def test_compute_rope_rejects_overflow(self) -> None:
        temporal, height, width = make_rope_freqs(24, 4, 8)
        positions = torch.tensor([[9, 0, 0]])
        with self.assertRaises(ValueError):
            compute_rope(positions, temporal, height, width)


if __name__ == "__main__":
    unittest.main()
