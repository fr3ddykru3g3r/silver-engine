"""Scientific invariants; synthetic fixtures only, no external outcomes."""
import io
import numpy as np
import pandas as pd
import pytest
from tools.audit_estimands_v1 import covariance_gap, safe_rows, metrics, mechanism, paired


def test_guard_rejects_horizon_touching_boundary_without_parsing_outcomes():
    data='issue_timestamp,standard_label,episode_id\n2025-09-09T00:00:00Z,DO_NOT_PARSE,SEALED\n'
    with pytest.raises(ValueError,match='protected horizon boundary'):
        safe_rows(io.StringIO(data),reject_any=True)


def test_guard_drops_protected_fields_and_does_not_return_counts():
    data='issue_timestamp,standard_label\n2025-09-08T00:00:00Z,0\n2025-09-10T00:00:00Z,not-a-label\n'
    rows=safe_rows(io.StringIO(data))
    assert rows==[{'issue_timestamp':'2025-09-08T00:00:00Z','standard_label':'0'}]


@pytest.mark.parametrize('date',['not-a-time','2025-09-08T00:00:00'])
def test_guard_rejects_ambiguous_time(date):
    with pytest.raises(ValueError):safe_rows(io.StringIO('issue_timestamp\n'+date+'\n'))


def test_finite_population_covariance_not_sample_covariance():
    n=np.array([1,3]);r=np.array([0.,1.])
    gap,rhs=covariance_gap(n,r)
    assert gap==pytest.approx(.25) and rhs==pytest.approx(gap)
    assert np.cov(n,r,ddof=1)[0,1]/n.mean()!=pytest.approx(gap)


def test_zero_covariance_is_not_independence():
    assert covariance_gap([1,2,3],[1,0,1])[0]==pytest.approx(0)


def test_no_positive_cluster_or_zero_multiplicity_fails():
    for n,r in [([],[]),([0,1],[1,0]),([1,np.nan],[1,0]),([1],[np.nan])]:
        with pytest.raises(ValueError):covariance_gap(n,r)


def test_arbitrary_base_weight_identity_and_scale_invariance():
    rng=np.random.default_rng(8)
    for _ in range(100):
        n=rng.uniform(.1,10,20);r=rng.normal(size=20);w=rng.uniform(0,3,20)
        gap,rhs=covariance_gap(n,r,w)
        assert gap==pytest.approx(rhs,abs=2e-14)
        assert gap==pytest.approx(covariance_gap(n,r,100*w)[0],abs=2e-14)


def test_tight_total_variation_bound_attainable_with_binary_alerts():
    n=np.array([1,2,8,9]);r=(n>n.mean()).astype(float)
    tv=abs(n/n.sum()-1/len(n)).sum()/2
    assert covariance_gap(n,r)[0]==pytest.approx(tv)
    assert covariance_gap(n,1-r)[0]==pytest.approx(-tv)


def test_model_ranking_can_reverse_without_refitting():
    n=[1,9];a=[1,0];b=[0,.8]
    episode_delta=np.mean(a)-np.mean(b)
    row_delta=np.average(a,weights=n)-np.average(b,weights=n)
    shift=covariance_gap(n,np.array(a)-b)[0]
    assert episode_delta>0 and row_delta<0
    assert row_delta==pytest.approx(episode_delta+shift)


def test_fpr_cancels_only_when_negative_measure_is_unchanged():
    # Same positive alerts; negative weighting changes FPR from .5 to .9.
    y=[1,1,0,0];a=[1,0,1,0];p=[.8,.2,.8,.2]
    row=metrics(y,a,p);other=metrics(y,a,p,[1,1,9,1])
    assert row['pod']==other['pod']
    assert row['tss']-other['tss']==pytest.approx(.4)


def fixture():
    # First episode: onset miss, two active hits. Second: onset hit only.
    return pd.DataFrame(dict(standard_label=[1,1,1,1,0,0],alert=[0,1,1,1,1,0],
        probability=[.2,.8,.7,.9,.8,.1],episode_id=['A','A','A','B','',''],
        eligibility=['ELIGIBLE_ONSET_POSITIVE','ALREADY_ACTIVE_PERSISTENCE','ALREADY_ACTIVE_PERSISTENCE',
                     'ELIGIBLE_ONSET_POSITIVE','ELIGIBLE_ONSET_NEGATIVE','ELIGIBLE_ONSET_NEGATIVE'],
        quiet_block=['','','','','Q1','Q2']))


def test_full_persistence_brier_auc_decomposition_and_any_alert_distinction():
    d=mechanism(fixture())
    assert d['maximum_algebra_error']<1e-14
    assert d['episode_any_alert_fraction']==1
    assert d['onset_detection_fraction']==.5
    assert d['mean_episode_window_detection']==pytest.approx(5/6)
    assert d['multiplicity_gap']==pytest.approx(-1/12)
    assert d['persistence_term']==pytest.approx(1/3)
    assert d['mapped_minus_onset']==pytest.approx(.25)


def test_persistence_term_undefined_without_onset_support_fails():
    f=fixture(); f.loc[3,'eligibility']='ALREADY_ACTIVE_PERSISTENCE'
    with pytest.raises(ValueError,match='one onset'):mechanism(f)


def test_matched_bootstrap_is_paired_and_identical_comparators_have_zero_delta():
    frames=[]
    for m in ['xgb_joint','xgb_no_proton','elastic_net_joint','elastic_net_no_proton','past_proton_ge10_proxy']:
        g=fixture();g['model']=m;frames.append(g)
    d=paired(pd.concat(frames),draws=100,seed=5)
    assert d['contrasts']['xgb__joint_minus_no_proton_onset']['interval95']==[0,0]
    assert d['positive_units']==2 and d['negative_units']==2
