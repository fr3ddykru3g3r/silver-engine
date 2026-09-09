"""Unit tests for the frozen downstream arm and exposure policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("train_matched_augmentation.py")
SPEC = importlib.util.spec_from_file_location("downstream_train", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def main() -> None:
    assert module.resolve_loss_policy("real", "none") == ("real", "none")
    assert module.resolve_loss_policy("real_weighted", "balanced") == (
        "real_weighted",
        "balanced",
    )
    for arm in ("duplicate", "synthetic"):
        assert module.resolve_loss_policy(arm, "none") == (arm, "none")

    invalid = [("real", "balanced"), ("real_weighted", "none"), ("duplicate", "balanced")]
    for arm, weighting in invalid:
        try:
            module.resolve_loss_policy(arm, weighting)
        except ValueError:
            pass
        else:
            raise AssertionError((arm, weighting))

    real = pd.DataFrame(
        [
            {"region_group_id": "g1", "label_m1plus_24h": 1, "sample_id": "a"},
            {"region_group_id": "g1", "label_m1plus_24h": 1, "sample_id": "b"},
            {"region_group_id": "g2", "label_m1plus_24h": 1, "sample_id": "c"},
            {"region_group_id": "g2", "label_m1plus_24h": 0, "sample_id": "d"},
        ]
    )
    duplicates = module.group_balanced_duplicates(real, 6, 2026)
    assert len(duplicates) == 6
    assert duplicates.sample_id.is_unique
    assert duplicates.region_group_id.value_counts().to_dict() == {"g1": 3, "g2": 3}
    assert duplicates.label_m1plus_24h.eq(1).all()
    print("downstream protocol self-test PASS")


if __name__ == "__main__":
    main()
