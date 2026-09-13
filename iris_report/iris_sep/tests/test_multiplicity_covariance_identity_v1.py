from __future__ import annotations

import numpy as np
import pandas as pd

from tools.verify_multiplicity_covariance_identity_v1 import mechanism_for_model


def make_rows(counts, fractions):
    rows=[]
    for i,(n,r) in enumerate(zip(counts,fractions)):
        detected=int(round(n*r))
        alerts=[1]*detected+[0]*(n-detected)
        for a in alerts:
            rows.append({"model":"m","standard_label":1,"episode_id":f"e{i}","alert":a})
    for a in [0,1,0,0]:
        rows.append({"model":"m","standard_label":0,"episode_id":None,"alert":a})
    return pd.DataFrame(rows)


def test_exact_covariance_identity_positive_inflation():
    # Longer episodes are easier: n=(1,4), r=(0,1), so row weighting inflates POD.
    d=make_rows([1,4],[0,1])
    r=mechanism_for_model(d,"m")
    assert abs(r["row_weighted_pod"]-0.8)<1e-12
    assert abs(r["episode_normalized_pod"]-0.5)<1e-12
    assert abs(r["row_minus_episode_pod"]-0.3)<1e-12
    assert r["covariance_multiplicity_detection_fraction"]>0
    assert r["identity_absolute_error"]<1e-12
    assert abs(r["row_minus_episode_tss"]-0.3)<1e-12


def test_identity_can_be_negative_when_long_episodes_are_harder():
    d=make_rows([1,4],[1,0])
    r=mechanism_for_model(d,"m")
    assert r["row_minus_episode_pod"]<0
    assert r["covariance_multiplicity_detection_fraction"]<0
    assert r["identity_absolute_error"]<1e-12


def test_equal_multiplicity_has_zero_mechanical_gap():
    d=make_rows([2,2],[0.5,1.0])
    r=mechanism_for_model(d,"m")
    assert abs(r["row_minus_episode_pod"])<1e-12
    assert abs(r["covariance_multiplicity_detection_fraction"])<1e-12
