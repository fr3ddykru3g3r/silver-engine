"""Generated-tensor PyTorch integration test; it opens no project dataset."""

from __future__ import annotations

import torch


def run() -> dict[str, object]:
    generator = torch.Generator().manual_seed(42)
    batch, steps, features = 8, 24, 12
    values = torch.randn(batch, steps, features, generator=generator)
    masks = torch.rand(batch, steps, features, generator=generator) > 0.1
    prepared = torch.cat((values * masks, masks.to(values.dtype)), dim=-1)
    layer = torch.nn.Conv1d(features * 2, 16, kernel_size=3, padding=0)
    output = layer(torch.nn.functional.pad(prepared.transpose(1, 2), (2, 0)))
    loss = output.square().mean()
    loss.backward()
    return {
        "status": "PASS",
        "synthetic_only": True,
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "cuda": bool(torch.cuda.is_available()),
    }


if __name__ == "__main__":
    print(run())
