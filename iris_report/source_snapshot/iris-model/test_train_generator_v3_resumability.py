from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

import train_generator_v3 as trainer


class ResumabilityTests(unittest.TestCase):
    def test_atomic_checkpoint_round_trip_restores_optimizer_and_history(self) -> None:
        model = torch.nn.Linear(2, 2)
        ema = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        records = pd.DataFrame({"region_group_id": ["RG1"], "sample_id": ["S1"]})
        args = Namespace(
            condition="base",
            seed=2026,
            diffusion_steps=20,
            base_channels=2,
            lambda_generic=0.08,
            lambda_hj=0.10,
            lambda_pil=0.10,
            physics_max_t_frac=0.20,
        )
        history = [{"step": 3, "loss": 1.25}]
        payload = trainer.checkpoint_payload(model, ema, optimizer, history, args, 3, torch.device("cpu"), records)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "latest.pt"
            trainer.atomic_torch_save(payload, checkpoint_path)
            restored_model = torch.nn.Linear(2, 2)
            restored_ema = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            step, restored_history, restored = trainer.load_resume(checkpoint_path, restored_model, restored_ema, restored_optimizer)
        self.assertEqual(step, 3)
        self.assertEqual(restored_history, history)
        self.assertEqual(restored["checkpoint_schema"], 2)
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))

    def test_heartbeat_is_atomic_and_reports_last_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            history = [{"step": 7, "loss": 0.5}]
            trainer.write_heartbeat(out, status="checkpointed", step=7, started=trainer.time.monotonic(), history=history)
            heartbeat = (out / "heartbeat.json").read_text()
        self.assertIn('"step": 7', heartbeat)
        self.assertIn('"status": "checkpointed"', heartbeat)


if __name__ == "__main__":
    unittest.main()
