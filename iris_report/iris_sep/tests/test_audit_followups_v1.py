import numpy as np
import pandas as pd
from tools.audit_followups_v1 import complete_subregimes, independent_controller


def test_incomplete_endpoint_does_not_erase_complete_source_run():
    months=[dict(year_month=f'2024-{m:02d}',complete_daily_coverage=m<12,prefixes=['sci'],versions=['3']) for m in range(1,13)]
    r=complete_subregimes(months)
    assert r==[dict(key='sci|3',start_month='2024-01',end_month='2024-11',months=11)]


def test_gap_and_version_change_split_source_runs():
    months=[dict(year_month=m,complete_daily_coverage=True,prefixes=['sci'],versions=[v])
            for m,v in [('2024-01','3'),('2024-03','3'),('2024-04','4')]]
    assert [r['months'] for r in complete_subregimes(months)]==[1,1,1]


def test_controller_current_and_unresolved_future_labels_do_not_affect_threshold():
    t=pd.Series(pd.date_range('2017-01-01',periods=60,tz='UTC'));y=np.zeros(60,int);p=np.linspace(.1,.9,60)
    a=independent_controller(t,y,p,.05,.8);y[40:]=1
    b=independent_controller(t,y,p,.05,.8)
    np.testing.assert_array_equal(a[:42],b[:42])
    # A discrete quantile need not change at the first newly resolved label.
    assert np.any(a[42:] != b[42:])


def test_controller_minimum_history_and_floor():
    t=pd.Series(pd.date_range('2017-01-01',periods=40,tz='UTC'));y=np.zeros(40,int);p=np.ones(40)*.01
    th=independent_controller(t,y,p,.05,.8)
    np.testing.assert_array_equal(th,np.full(40,.05))
