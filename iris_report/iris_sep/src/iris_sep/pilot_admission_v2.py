"""Source-era, source-revision, and magnitude-aware pilot admission."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Mapping, Any

import numpy as np

from iris_report.iris_sep.src.iris_sep.pilot_replay import replay_forecast


@dataclass(frozen=True)
class AdmissionPolicyV2:
    supported_from: datetime
    supported_through: datetime
    allowed_source_revisions: Mapping[str, tuple[str, ...]]
    maximum_abs_standardized_feature: float

    def __post_init__(self) -> None:
        if (self.supported_from.tzinfo is None or self.supported_through.tzinfo is None or
                self.supported_from >= self.supported_through):
            raise ValueError("ordered timezone-aware support interval required")
        if not math.isfinite(self.maximum_abs_standardized_feature) or self.maximum_abs_standardized_feature <= 0:
            raise ValueError("positive finite magnitude boundary required")


def replay_forecast_v2(*, admission_policy: AdmissionPolicyV2,
                       transformed_features: Any, model_outputs: Any, **request):
    reasons = list(request.pop("abstention_reasons", ()))
    issue = request["issued_at"]
    if issue < admission_policy.supported_from or issue > admission_policy.supported_through:
        reasons.append("UNSUPPORTED_SOURCE_ERA")
    freshness = request["data_freshness"]
    for modality, allowed in admission_policy.allowed_source_revisions.items():
        if modality in request["missing_modalities"]:
            continue
        revision = freshness.get(modality, {}).get("source_revision")
        if not allowed or revision not in allowed:
            reasons.append("SOURCE_REVISION_FAILURE")
    try:
        features = np.asarray(transformed_features, dtype=float)
        outputs = np.asarray(model_outputs, dtype=float)
        if features.size == 0 or outputs.size == 0:
            raise ValueError("empty diagnostic input")
        maximum = float(np.max(np.abs(features)))
        finite = bool(np.isfinite(features).all() and np.isfinite(outputs).all())
    except (TypeError, ValueError):
        maximum, finite = math.inf, False
    if not finite or maximum > admission_policy.maximum_abs_standardized_feature:
        reasons.append("OUT_OF_DISTRIBUTION")
    request["abstention_reasons"] = sorted(set(reasons))
    return replay_forecast(**request)
