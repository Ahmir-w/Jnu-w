import sys
from pathlib import Path
from datetime import datetime
import unittest
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fedmr_2026.federated import (FederatedCoordinator, ClientUpdate, perturb_update,
    train_client_update, run_simulated_federation, memory_features, influence_from_features)
from fedmr_2026.memory import EnhancedMemoryUnit, DynamicMemoryBank
from fedmr_2026.selection import SubsetSelector
from fedmr_2026.inference import EnhancedInferenceEngine
from fedmr_2026.config import ExperimentConfig


def units():
    return [EnhancedMemoryUnit(np.array([i/10, 1.0, -0.2], dtype=np.float32),
             str(i), datetime.now(), 'test', difficulty=i/10, retrieval_freq=i) for i in range(8)]


class PrivacyTest(unittest.TestCase):
    def test_clip_and_zero_noise(self):
        x, d = perturb_update(np.array([3., 4.]), 1., 0.)
        np.testing.assert_allclose(x, [0.6, 0.8], atol=1e-7)
        self.assertAlmostEqual(d['clipped_l2'], 1.)
        self.assertTrue(d['was_clipped'])
        x, d = perturb_update(np.zeros(3), 1., 0.)
        self.assertFalse(d['was_clipped'])
        np.testing.assert_equal(x, np.zeros(3))

    def test_noise(self):
        x, _ = perturb_update(np.zeros(100000), 1., .001, np.random.default_rng(42))
        self.assertLess(abs(x.mean()), 0.00002)
        self.assertAlmostEqual(float(x.std()), .001, delta=.00002)
        a, _ = perturb_update(np.zeros(100), 1., .001)
        b, _ = perturb_update(np.zeros(100), 1., .001)
        self.assertFalse(np.array_equal(a,b))

    def test_weighted_aggregation(self):
        f = FederatedCoordinator(8, 4)
        old = f.broadcast()
        f.aggregate([ClientUpdate(0, 1, np.ones_like(old)),ClientUpdate(0,3,np.full_like(old,3))])
        np.testing.assert_allclose(f.broadcast(),old+2.5,atol=1e-6)
        with self.assertRaises(ValueError):
            f.aggregate([ClientUpdate(0,1,np.ones_like(old))])

    def test_train_and_save(self):
        f = FederatedCoordinator(8,4)
        u = units()
        update, diag = train_client_update(u, f.broadcast(), 0, 8, 4,C=.01,sigma=.001,local_epochs=1)
        self.assertLessEqual(diag['clipped_l2'], .0100001)
        self.assertEqual(set(update.__dict__), {'round_id','sample_count','delta'})
        f.aggregate([update])
        self.assertTrue(np.isfinite(f.local_scores(u)).all())
        self.assertFalse(hasattr(f,'global_units'))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'global.npz'
            f.save(path)
            f2 = FederatedCoordinator.load(path)
            np.testing.assert_equal(f.broadcast(),f2.broadcast())
            self.assertEqual(f2.round_id,1)

    def test_integrated_scoring_and_local_retrieval(self):
        u = units()
        f = run_simulated_federation(u,num_clients=4,rounds=2,hidden_dim=4,local_epochs=1)
        bank=DynamicMemoryBank()
        for x in u: bank.add_unit(x)
        s=SubsetSelector(bank,f)
        self.assertEqual(s.mu,.4)
        self.assertAlmostEqual(s.compute_federated_score(u[0]),.6*s.compute_local_value(u[0])+.4*f.get_global_score(u[0]),places=5)
        bank.current_subset_indices=s.offline_selection(3,0,0,u[0].key)
        e=EnhancedInferenceEngine(bank,f)
        e.update_models()
        result=e.knn_search_local(u[0].key)
        self.assertEqual(len(result),3)
        self.assertTrue({t for t,_ in result}.issubset({x.target for x in bank.get_subset_units()}))
        self.assertEqual(ExperimentConfig(Path('a'),Path('b')).mu,.4)

if __name__=='__main__': unittest.main()
