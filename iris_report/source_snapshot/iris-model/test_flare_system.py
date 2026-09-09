from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from flare_system.data import FeatureScaler, select_group_subset, select_label_blind_group_subset
from flare_system.aarp import decode_fixed_width, parse_aarp_id


class DataContractTests(unittest.TestCase):
    def sample_frame(self) -> pd.DataFrame:
        rows = []
        for group in ("A", "B"):
            for index in range(10):
                rows.append(
                    {
                        "sample_id": f"{group}{index}",
                        "region_group_id": group,
                        "t_rec": f"2020-01-{index + 1:02d}T00:00:00Z",
                        "label_m1plus_24h": int(index in {2, 7}),
                        "usflux": 10 ** (index + 1),
                        "r_value": index + 1,
                        "latitude_deg": index - 5,
                        "cmd_deg": index * 2,
                    }
                )
        return pd.DataFrame(rows)

    def test_group_subset_is_balanced_deterministic_and_bounded(self):
        frame = self.sample_frame()
        first = select_group_subset(frame, per_group=4, positive_cap=2, seed=9)
        second = select_group_subset(frame, per_group=4, positive_cap=2, seed=9)
        self.assertEqual(first.sample_id.tolist(), second.sample_id.tolist())
        self.assertEqual(first.groupby("region_group_id").size().to_dict(), {"A": 4, "B": 4})
        self.assertEqual(first.groupby("region_group_id").label_m1plus_24h.sum().to_dict(), {"A": 2, "B": 2})

    def test_aarp_fixed_width_identity_contract(self):
        encoded = np.array(
            [[bytes([byte]) for byte in b"2014.06.14_15:48:00_7h@1h_4228.fits".ljust(64, b"\x00")]],
            dtype="S1",
        )
        decoded = decode_fixed_width(encoded)
        self.assertEqual(decoded, ["2014.06.14_15:48:00_7h@1h_4228.fits"])
        identity = parse_aarp_id(decoded[0])
        self.assertEqual(identity.issue_time_utc, "2014-06-14T15:48:00Z")
        self.assertEqual(identity.aarp_number, 4228)

    def test_feature_scaler_is_train_fitted_and_finite(self):
        frame = self.sample_frame()
        scaler = FeatureScaler.fit(frame)
        transformed = scaler.transform(frame)
        self.assertEqual(transformed.shape, (20, 4))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertLessEqual(float(np.abs(transformed).max()), 8.0)

    def test_locked_test_selection_is_label_blind(self):
        frame = self.sample_frame()
        first = select_label_blind_group_subset(frame, per_group=3, seed=11)
        relabeled = frame.copy()
        relabeled["label_m1plus_24h"] = 1 - relabeled["label_m1plus_24h"]
        second = select_label_blind_group_subset(relabeled, per_group=3, seed=11)
        self.assertEqual(first.sample_id.tolist(), second.sample_id.tolist())


try:
    import torch
    from flare_system.model import HybridFlareNet, magnetogram_channels

    class ModelContractTests(unittest.TestCase):
        def test_channel_and_model_shapes(self):
            x = torch.randn(2, 1, 64, 64)
            self.assertEqual(tuple(magnetogram_channels(x).shape), (2, 3, 64, 64))
            model = HybridFlareNet(width=16)
            output = model(x, torch.randn(2, 4))
            self.assertEqual(tuple(output["logit"].shape), (2,))
            self.assertEqual(tuple(output["aux_physics"].shape), (2, 2))
except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
