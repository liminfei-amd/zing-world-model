from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WEIGHTS = Path(os.environ.get("ZING_PRETRAINED_DIR", str(Path.home() / ".cache/zing-0.5/pretrained")))


@unittest.skipUnless(
    WEIGHTS.is_dir() and (WEIGHTS / "vae").is_dir(),
    "Zing VAE weights are not on this host",
)
class StreamVaeMatchesFullDecode(unittest.TestCase):
    def test_chunked_decode_matches_full_decode(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("HIP/CUDA GPU required")
        from zing_v0_5.model.vae import WanVAE

        vae = WanVAE(WEIGHTS)
        vae.eval().to(device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(0)
        latents = torch.randn(1, 5, 48, 30, 52, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            full = vae.decode(latents)
            vae.begin_stream()
            first = vae.decode_chunk(latents[:, :1])
            rest = vae.decode_chunk(latents[:, 1:])
            vae.end_stream()
            streamed = torch.cat([first, rest], dim=1)
        self.assertEqual(tuple(full.shape), tuple(streamed.shape))
        self.assertTrue(torch.equal(full, streamed))


if __name__ == "__main__":
    unittest.main()
