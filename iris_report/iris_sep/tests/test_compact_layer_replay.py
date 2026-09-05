import unittest
import numpy as np
import torch
from iris_report.iris_sep.src.iris_sep.modeling.compact_layer_replay import replay_model_layers, transform_from_preprocessing
from iris_report.iris_sep.src.iris_sep.modeling.tabular_multibranch import BranchInput, IRISSEPTabularModel, TabularModelConfig


class ReplayTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(1)
        return IRISSEPTabularModel(TabularModelConfig(2,2,1,dropout=0.0,missing_modality_dropout=0.0))

    def inputs(self):
        return {
            'magnetic':BranchInput(torch.ones(2,2),torch.ones(2,2,dtype=torch.bool)),
            'eruption':BranchInput(torch.zeros(2,2),torch.ones(2,2,dtype=torch.bool)),
            'particle_context':BranchInput(torch.zeros(2,1),torch.ones(2,1,dtype=torch.bool))}

    def test_all_missing_behavior_is_finite(self):
        model=self.model();inputs={
            'magnetic':BranchInput(torch.zeros(3,2),torch.zeros(3,2,dtype=torch.bool)),
            'eruption':BranchInput(torch.zeros(3,2),torch.zeros(3,2,dtype=torch.bool)),
            'particle_context':BranchInput(torch.zeros(3,1),torch.zeros(3,1,dtype=torch.bool))}
        result=replay_model_layers(model,inputs);self.assertEqual(result['status'],'FINITE_REPLAY');self.assertEqual(result['all_missing_rows'],3);self.assertTrue(result['forward_replay_exact'])
        self.assertEqual(result['checkpoint_parameter_audit']['nonfinite_parameter_tensors'],0)

    def test_first_nonfinite_layer_is_identified_without_guessing_cause(self):
        model=self.model()
        with torch.no_grad(): model.branches['magnetic'].network[0].weight.fill_(3e38)
        result=replay_model_layers(model,self.inputs());self.assertEqual(result['status'],'NONFINITE_REPRODUCED');self.assertEqual(result['first_nonfinite_stage'],'magnetic.branch.0.Linear')
        self.assertEqual(result['checkpoint_parameter_audit']['nonfinite_parameter_tensors'],0)

    def test_nonfinite_checkpoint_parameter_is_reported_separately(self):
        model=self.model()
        with torch.no_grad(): model.branches['magnetic'].network[0].weight[0,0]=float('nan')
        result=replay_model_layers(model,self.inputs())
        self.assertEqual(result['status'],'NONFINITE_REPRODUCED')
        self.assertEqual(result['first_nonfinite_stage'],'magnetic.branch.0.Linear')
        audit=result['checkpoint_parameter_audit']
        self.assertEqual(audit['nonfinite_parameter_tensors'],1)
        self.assertEqual(audit['nonfinite_parameter_values'],1)
        self.assertEqual(audit['first_nonfinite_parameter'],'branches.magnetic.network.0.weight')

    def test_nonbinary_missing_mask_is_rejected(self):
        model=self.model();inputs=self.inputs();inputs['magnetic']=BranchInput(torch.ones(2,2),torch.tensor([[1.0,.5],[1.0,1.0]]))
        with self.assertRaisesRegex(ValueError,'feature mask'):
            replay_model_layers(model,inputs)

    def test_train_fitted_transform_reports_float32_cast_overflow(self):
        preprocessing={'fit_role':'train','modalities':{
            'magnetic':{'columns':['m'],'always_unavailable':False,'median':[0.0],'mean':[0.0],'scale':[1e-8]},
            'eruption':{'columns':['e'],'always_unavailable':False,'median':[0.0],'mean':[0.0],'scale':[1.0]},
            'particle_context':{'columns':[],'placeholder_width':1,'always_unavailable':True}}}
        raw={'magnetic':np.array([[1e32]]),'eruption':np.array([[0.0]])}
        values,masks,audit=transform_from_preprocessing(raw,preprocessing);self.assertEqual(audit['modalities']['magnetic']['pre_cast_nonfinite'],0)
        self.assertEqual(audit['modalities']['magnetic']['post_cast_nonfinite'],1);self.assertFalse(np.isfinite(values['magnetic']).all())

    def test_train_fitted_transform_reports_zero_feature_support(self):
        preprocessing={'fit_role':'train','modalities':{
            'magnetic':{'columns':['m1','m2'],'always_unavailable':False,'median':[0.0,1.0],'mean':[0.0,1.0],'scale':[1.0,1.0]},
            'eruption':{'columns':['e'],'always_unavailable':False,'median':[0.0],'mean':[0.0],'scale':[1.0]},
            'particle_context':{'columns':[],'placeholder_width':1,'always_unavailable':True}}}
        raw={'magnetic':np.array([[1.0,np.nan],[2.0,np.nan]]),'eruption':np.array([[0.0],[1.0]])}
        _,_,audit=transform_from_preprocessing(raw,preprocessing)
        self.assertEqual(audit['modalities']['magnetic']['observed_per_feature'],[2,0])
        self.assertEqual(audit['modalities']['magnetic']['zero_support_features'],1)

if __name__=='__main__': unittest.main()
