"""Independent, calibrated real-data flare forecasting pipeline.

This package is intentionally separate from the synthetic generator gate.  The
locked synthetic-ablation protocol can fail closed without preventing a valid
real-only benchmark from being trained and evaluated.
"""

from .data import FeatureScaler, SamplingConfig, build_selected_records

__all__ = ["FeatureScaler", "SamplingConfig", "build_selected_records"]
