from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from zing_v0_5.model.kv_cache import CausalKVCache


class WindowedKVStaysBounded(unittest.TestCase):
    def _run_blocks(self, cache: CausalKVCache, blocks: int, rope_limit: int | None = None) -> None:
        if rope_limit is not None:
            cache.rope_max_seq_len = rope_limit
        height, width = 4, 5
        tokens = height * width
        cache.allocate_window(tokens, num_heads=2, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
        for block in range(blocks):
            frames = 1 if block == 0 else cache.frames_per_block
            grid = torch.tensor([[frames, height, width]])
            positions, block_id, _ = cache.prepare(grid, torch.device("cpu"), None, False)
            sequence = frames * tokens
            keys = [
                (torch.randn(1, sequence, 2, 4), torch.randn(1, sequence, 2, 4))
                for _ in range(cache.num_layers)
            ]
            cache.update(keys, [None] * cache.num_layers, positions, block_id, "final", None)

    def test_used_tokens_plateau_and_storage_stays_preallocated(self) -> None:
        cache = CausalKVCache(
            num_layers=2,
            action_history_frames=4,
            local_attn_size=33,
            sink_size=5,
            frames_per_block=4,
        )
        self._run_blocks(cache, blocks=16)
        self.assertEqual(cache.self_k[0].shape[1], cache.kv_capacity)
        plateau = cache.used_tokens
        self.assertGreater(plateau, 0)
        self.assertLessEqual(plateau, cache.kv_capacity)
        height, width = 4, 5
        tokens = height * width
        for _ in range(16):
            frames = cache.frames_per_block
            grid = torch.tensor([[frames, height, width]])
            positions, block_id, _ = cache.prepare(grid, torch.device("cpu"), None, False)
            sequence = frames * tokens
            keys = [
                (torch.randn(1, sequence, 2, 4), torch.randn(1, sequence, 2, 4))
                for _ in range(cache.num_layers)
            ]
            cache.update(keys, [None] * cache.num_layers, positions, block_id, "final", None)
        self.assertEqual(cache.self_k[0].shape[1], cache.kv_capacity)
        self.assertLessEqual(cache.used_tokens, cache.kv_capacity)
        self.assertLessEqual(abs(cache.used_tokens - plateau), tokens * cache.frames_per_block)

    def test_temporal_positions_stay_inside_rope_table(self) -> None:
        cache = CausalKVCache(
            num_layers=1,
            action_history_frames=4,
            local_attn_size=33,
            sink_size=5,
            frames_per_block=4,
            rope_max_seq_len=64,
        )
        self._run_blocks(cache, blocks=40, rope_limit=64)
        self.assertIsNotNone(cache.positions)
        self.assertLess(int(cache.positions[:, 0].max()), 64)
        self.assertGreaterEqual(int(cache.positions[:, 0].min()), 0)

    def test_stage_current_matches_concatenated_history(self) -> None:
        cache = CausalKVCache(
            num_layers=1,
            action_history_frames=4,
            local_attn_size=33,
            sink_size=5,
            frames_per_block=4,
        )
        self._run_blocks(cache, blocks=3)
        used = cache.used_tokens
        current = torch.randn(1, 20, 2, 4)
        history = cache.history(0)[0]
        staged = cache.stage_current(0, current, current)
        self.assertIsNotNone(staged)
        raw_key, _ = staged
        expected = torch.cat((history, current), dim=1)
        self.assertEqual(tuple(raw_key.shape), tuple(expected.shape))
        self.assertTrue(torch.equal(raw_key, expected))
        self.assertEqual(cache.used_tokens, used)


if __name__ == "__main__":
    unittest.main()
