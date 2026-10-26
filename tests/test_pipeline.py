import unittest
import numpy as np
from scipy.special import softmax
from visionguard.pipeline import splits,neural_logits,select_threshold,metrics,corrupt,temperature_fit

class VisionTests(unittest.TestCase):
    def test_disjoint_stratified_splits(self):
        y=np.tile(np.arange(10),100);parts=splits(y)
        self.assertEqual(len(np.unique(np.concatenate(parts))),len(y))
        for part in parts:self.assertEqual(set(y[part]),set(range(10)))
    def test_neural_forward_known_weights(self):
        actual=neural_logits([[1,2]], [np.eye(2),np.eye(2)],[np.array([-2,0]),np.array([.5,-.5])])
        np.testing.assert_allclose(actual,[[.5,1.5]])
    def test_temperature_probability_validity(self):
        logits=np.eye(10)*8;t=temperature_fit(logits,np.arange(10))
        self.assertTrue(.2<=t<=5);np.testing.assert_allclose(softmax(logits/t,axis=1).sum(axis=1),1)
    def test_abstention_when_no_validation_support(self):
        p=np.tile([.6,.4],(10,1));self.assertGreater(select_threshold(np.ones(10),p),1)
    def test_zero_coverage_is_null(self):
        report=metrics(np.arange(10),np.eye(10),1.1)
        self.assertEqual(report['coverage'],0);self.assertIsNone(report['selective_accuracy'])
    def test_corruptions_are_bounded_and_do_not_mutate(self):
        x=np.full((3,64),.5)
        for kind in ['noise','occlusion','shift']:
            a=corrupt(x,kind);self.assertEqual(a.shape,x.shape);self.assertTrue(((a>=0)&(a<=1)).all())
        np.testing.assert_array_equal(x,.5)
    def test_shift_has_zero_padding_not_wraparound(self):
        x=np.ones((1,64));shift=corrupt(x,'shift').reshape(8,8)
        np.testing.assert_array_equal(shift[:,0],0);np.testing.assert_array_equal(shift[:,1:],1)
    def test_threshold_selected_without_test(self):
        p=np.tile([.99,.01],(40,1));self.assertAlmostEqual(select_threshold(np.zeros(40),p),.99)

if __name__=='__main__':unittest.main()
