import unittest
import pandas as pd
from iris_report.iris_sep.tools.run_inner_training_diagnostic import folds, TARGET

class FoldTests(unittest.TestCase):
    def frame(self):
        times=pd.date_range('1990-01-01',periods=600,freq='D',tz='UTC')
        return pd.DataFrame({'role':'train','issue_id':[str(i) for i in range(600)],'unit_id':[str(i//2) for i in range(600)],'window_end':times.astype(str),TARGET:[(i//2)%2 for i in range(600)]})
    def test_strict_purge_and_coverage(self):
        f=self.frame(); ff=folds(f)
        for fold in ff:
            previous=None;seen=set()
            for ix in fold['indices'].values():
                g=f.iloc[ix];units=set(g.unit_id);self.assertFalse(seen & units);seen|=units
                if previous is not None:self.assertGreater(pd.to_datetime(g.window_end,utc=True).min(),previous+pd.Timedelta(hours=24))
                previous=pd.to_datetime(g.window_end,utc=True).max()
            self.assertEqual(len(seen|set(fold['purged_units'])),fold['prefix_units'])
    def test_forbidden_outer_role(self):
        f=self.frame();f.loc[0,'role']='validation_monitor'
        with self.assertRaises(ValueError):folds(f)
    def test_duplicate_issue_rejected(self):
        f=self.frame();f.loc[1,'issue_id']=f.loc[0,'issue_id']
        with self.assertRaises(ValueError):folds(f)
if __name__=='__main__':unittest.main()
