from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from zing_v0_5.model import attention


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
    for query_length, key_length in zip(
        query_lengths.tolist(), key_lengths.tolist(), strict=True
    ):
        sample_query = query[query_offset : query_offset + query_length].transpose(0, 1).float()
        sample_key = key[key_offset : key_offset + key_length].transpose(0, 1).float()
        sample_value = value[key_offset : key_offset + key_length].transpose(0, 1).float()
        scores = torch.matmul(sample_query, sample_key.transpose(-1, -2)) * scale
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(weights, sample_value).transpose(0, 1).to(query.dtype))
        query_offset += query_length
        key_offset += key_length
    return torch.cat(outputs, dim=0)


def _inputs(
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    heads: int = 2,
    dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(int(query_lengths.sum()), heads, dim)
    key = torch.randn(int(key_lengths.sum()), heads, dim)
    value = torch.randn(int(key_lengths.sum()), heads, dim)
    return query, key, value


class SdpaVarlenFallbackTests(unittest.TestCase):
    def _assert_matches_naive(
        self,
        query_lengths: torch.Tensor,
        key_lengths: torch.Tensor,
    ) -> None:
        query, key, value = _inputs(query_lengths, key_lengths)
        with mock.patch.dict(os.environ, {"ZING_ATTENTION_BACKEND": "sdpa-math"}):
            actual = attention.flash_attention_varlen(
                query,
                key,
                value,
                query_lengths,
                key_lengths,
                deterministic=False,
            )
        expected = _naive_varlen(query, key, value, query_lengths, key_lengths)
        self.assertTrue(
            torch.allclose(actual.float(), expected.float(), atol=1e-5, rtol=1e-4)
        )

    def test_equal_lengths_match_naive_attention(self) -> None:
        torch.manual_seed(0)
        self._assert_matches_naive(
            torch.tensor([4, 4], dtype=torch.int32),
            torch.tensor([6, 6], dtype=torch.int32),
        )

    def test_unequal_lengths_match_naive_attention(self) -> None:
        torch.manual_seed(1)
        self._assert_matches_naive(
            torch.tensor([3, 5], dtype=torch.int32),
            torch.tensor([4, 2], dtype=torch.int32),
        )

    def test_auto_preserves_flash_attention_when_available(self) -> None:
        query_lengths = torch.tensor([2], dtype=torch.int32)
        key_lengths = torch.tensor([2], dtype=torch.int32)
        query, key, value = _inputs(query_lengths, key_lengths)
        calls = []

        def fake_flash(*args, **kwargs):
            calls.append((args, kwargs))
            return query

        with (
            mock.patch.object(attention, "_flash_attn_varlen_func", fake_flash),
            mock.patch.object(torch.version, "hip", None),
            mock.patch.dict(os.environ, {"ZING_ATTENTION_BACKEND": "auto"}),
        ):
            actual = attention.flash_attention_varlen(
                query,
                key,
                value,
                query_lengths,
                key_lengths,
                deterministic=False,
            )

        self.assertIs(actual, query)
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertTrue(torch.equal(args[3], torch.tensor([0, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[4], torch.tensor([0, 2], dtype=torch.int32)))
        self.assertEqual(args[5:7], (2, 2))
        self.assertEqual(
            kwargs,
            {
                "dropout_p": 0.0,
                "softmax_scale": None,
                "causal": False,
                "deterministic": False,
            },
        )

    def test_auto_falls_back_to_sdpa_when_flash_is_unavailable(self) -> None:
        torch.manual_seed(2)
        query_lengths = torch.tensor([2, 2], dtype=torch.int32)
        key_lengths = torch.tensor([3, 3], dtype=torch.int32)
        query, key, value = _inputs(query_lengths, key_lengths)
        expected = _naive_varlen(query, key, value, query_lengths, key_lengths)

        with (
            mock.patch.object(attention, "_flash_attn_varlen_func", None),
            mock.patch.dict(os.environ, {"ZING_ATTENTION_BACKEND": "auto"}),
        ):
            actual = attention.flash_attention_varlen(
                query,
                key,
                value,
                query_lengths,
                key_lengths,
                deterministic=False,
            )

        self.assertTrue(
            torch.allclose(actual.float(), expected.float(), atol=1e-5, rtol=1e-4)
        )

    def test_auto_does_not_select_cuda_flash_attention_on_rocm(self) -> None:
        with (
            mock.patch.object(attention, "_flash_attn_varlen_func", mock.Mock()),
            mock.patch.object(torch.version, "hip", "test-hip"),
            mock.patch.dict(os.environ, {"ZING_ATTENTION_BACKEND": "auto"}),
        ):
            self.assertEqual(attention._resolve_attention_backend(), "sdpa-math")

    def test_explicit_flash_fails_when_package_is_unavailable(self) -> None:
        query_lengths = torch.tensor([1], dtype=torch.int32)
        query, key, value = _inputs(query_lengths, query_lengths)
        with (
            mock.patch.object(attention, "_flash_attn_varlen_func", None),
            mock.patch.dict(os.environ, {"ZING_ATTENTION_BACKEND": "flash"}),
            self.assertRaisesRegex(ImportError, "requires the CUDA flash-attn package"),
        ):
            attention.flash_attention_varlen(
                query,
                key,
                value,
                query_lengths,
                query_lengths,
                deterministic=False,
            )

    def test_invalid_backend_is_rejected(self) -> None:
        query_lengths = torch.tensor([1], dtype=torch.int32)
        query, key, value = _inputs(query_lengths, query_lengths)
        with (
            mock.patch.dict(os.environ, {"ZING_ATTENTION_BACKEND": "unknown"}),
            self.assertRaisesRegex(ValueError, "unsupported ZING_ATTENTION_BACKEND"),
        ):
            attention.flash_attention_varlen(
                query,
                key,
                value,
                query_lengths,
                query_lengths,
                deterministic=False,
            )

    def test_packed_token_counts_must_match_lengths(self) -> None:
        query_lengths = torch.tensor([2], dtype=torch.int32)
        key_lengths = torch.tensor([2], dtype=torch.int32)
        query, key, value = _inputs(query_lengths, key_lengths)
        with (
            mock.patch.dict(os.environ, {"ZING_ATTENTION_BACKEND": "sdpa-math"}),
            self.assertRaisesRegex(ValueError, "packed token counts"),
        ):
            attention.flash_attention_varlen(
                query[:-1],
                key,
                value,
                query_lengths,
                key_lengths,
                deterministic=True,
            )


if __name__ == "__main__":
    unittest.main()
