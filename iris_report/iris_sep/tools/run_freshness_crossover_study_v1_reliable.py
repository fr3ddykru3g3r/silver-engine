"""Transport-only wrapper for the frozen freshness crossover study.

Adds bounded retries for transient archive connection failures, then executes the
unchanged direct study runner. It does not alter URLs, data, labels, features,
models, thresholds, roles, delay conditions, or evaluation rules.
"""
from __future__ import annotations
import runpy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_OriginalSession=requests.Session

class ReliableSession(_OriginalSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        retry=Retry(
            total=7,
            connect=7,
            read=7,
            status=7,
            backoff_factor=1.0,
            status_forcelist=(429,500,502,503,504),
            allowed_methods=frozenset({'GET'}),
            raise_on_status=False,
        )
        adapter=HTTPAdapter(max_retries=retry,pool_connections=4,pool_maxsize=4)
        self.mount('https://',adapter)
        self.mount('http://',adapter)

requests.Session=ReliableSession
runpy.run_module('iris_report.iris_sep.tools.run_freshness_crossover_study_v1',run_name='__main__')
