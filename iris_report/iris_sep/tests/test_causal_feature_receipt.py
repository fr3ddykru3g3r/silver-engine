from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from iris_report.iris_sep.src.iris_sep.causal_feature_receipt import (
    CausalFeatureReceiptError,
    build_causal_feature_derivation_receipt,
    validate_causal_feature_derivation_receipt,
)
from iris_report.iris_sep.src.iris_sep.source_authentication import (
    authenticate_acquisition_receipts,
    build_acquisition_receipt,
)


ISSUE = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
FEATURES = {"SOLAR": ("s1",), "XRS": ("x1",), "PROTON": ("p1",)}
ORDER = ("SOLAR", "XRS", "PROTON")


def acquisitions():
    rows = []
    for source_id, url, digest in (
        (
            "NOAA_SWPC_PRIMARY_XRS_7D",
            "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
            "a" * 64,
        ),
        (
            "NOAA_SWPC_PRIMARY_PROTON_7D",
            "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json",
            "b" * 64,
        ),
    ):
        rows.append(
            build_acquisition_receipt(
                source_id=source_id,
                retrieved_at=ISSUE - timedelta(seconds=30),
                artifact_sha256=digest,
                artifact_bytes=100,
                source_url=url,
            )
        )
    return rows


def auth(rows):
    return authenticate_acquisition_receipts(
        issue_time=ISSUE,
        receipts=rows,
        required_source_ids=[
            "NOAA_SWPC_PRIMARY_XRS_7D",
            "NOAA_SWPC_PRIMARY_PROTON_7D",
        ],
    )


class CausalFeatureReceiptTests(unittest.TestCase):
    def test_exact_row_and_source_chain_recomputes(self):
        rows = acquisitions()
        authentication = auth(rows)
        frame = pd.DataFrame([[1.0, float("nan"), 3.0]], columns=["s1", "x1", "p1"])
        receipt = build_causal_feature_derivation_receipt(
            issue_time=ISSUE,
            feature_row=frame,
            feature_families=FEATURES,
            family_order=ORDER,
            source_authentication=authentication,
            acquisition_receipts=rows,
        )
        rebuilt = validate_causal_feature_derivation_receipt(
            receipt=receipt,
            issue_time=ISSUE,
            feature_row=frame,
            feature_families=FEATURES,
            family_order=ORDER,
            source_authentication=authentication,
            acquisition_receipts=rows,
        )
        self.assertEqual(rebuilt, receipt)
        self.assertEqual(receipt["feature_count"], 3)
        self.assertFalse(receipt["retrospective_imputation_used"])

    def test_feature_value_tamper_breaks_chain(self):
        rows = acquisitions()
        authentication = auth(rows)
        frame = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["s1", "x1", "p1"])
        receipt = build_causal_feature_derivation_receipt(
            issue_time=ISSUE,
            feature_row=frame,
            feature_families=FEATURES,
            family_order=ORDER,
            source_authentication=authentication,
            acquisition_receipts=rows,
        )
        tampered = frame.copy()
        tampered.loc[0, "x1"] = 9.0
        with self.assertRaisesRegex(CausalFeatureReceiptError, "does not match recomputation"):
            validate_causal_feature_derivation_receipt(
                receipt=receipt,
                issue_time=ISSUE,
                feature_row=tampered,
                feature_families=FEATURES,
                family_order=ORDER,
                source_authentication=authentication,
                acquisition_receipts=rows,
            )

    def test_unauthenticated_source_chain_is_rejected(self):
        rows = acquisitions()
        authentication = auth(rows)
        authentication["authenticated_for_prospective_use"] = False
        frame = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["s1", "x1", "p1"])
        with self.assertRaisesRegex(CausalFeatureReceiptError, "not authenticated"):
            build_causal_feature_derivation_receipt(
                issue_time=ISSUE,
                feature_row=frame,
                feature_families=FEATURES,
                family_order=ORDER,
                source_authentication=authentication,
                acquisition_receipts=rows,
            )


if __name__ == "__main__":
    unittest.main()
