#!/usr/bin/env python3
"""Unit tests for the detectors, the baselines and the evaluation helpers."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.evaluation import (cost_model_abort, cost_model_skip, fast_auprc, fast_auroc,  # noqa: E402
                            holm_bonferroni, rolling_threshold_predictions, stratified_block_bootstrap)
from src.models.autoencoder import TelemetryAutoencoder  # noqa: E402
from src.models.baselines import (GradientBoostingBaseline, PreviousOutcomeBaseline,  # noqa: E402
                                  RandomForestBaseline, RollingFailureRateBaseline)
from src.models.classical_ad import IsolationForestAD, PCAReconstructionAD  # noqa: E402


class TestModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.D = 12
        self.X_train_normal = rng.normal(0, 1.0, (300, self.D))
        # One constant column among passing builds, non constant at test time.
        self.X_train_normal[:, 3] = 0.0
        self.X_val_normal = rng.normal(0, 1.0, (60, self.D))
        self.X_val_normal[:, 3] = 0.0
        X_test_normal = rng.normal(0, 1.0, (40, self.D))
        X_test_normal[:, 3] = 0.0
        X_test_anom = rng.normal(3.0, 1.5, (10, self.D))
        self.X_test = np.vstack([X_test_normal, X_test_anom])
        self.y_test = np.array([0] * 40 + [1] * 10)

    def test_autoencoder_ignores_zero_variance_feature(self):
        ae = TelemetryAutoencoder(hidden_layer_sizes=(16, 8, 16), max_iter=150, random_state=42)
        ae.fit(self.X_train_normal)
        self.assertFalse(ae.active_features[3])
        self.assertEqual(ae.feature_weights[3], 0.0)
        ae.calibrate_threshold(self.X_val_normal, percentile=90.0)
        scores = ae.predict_score(self.X_test)
        self.assertGreater(scores[40:].mean(), scores[:40].mean())
        attr = ae.compute_feature_attribution(self.X_test)
        self.assertEqual(attr.shape, (50, self.D))
        share = attr.sum(axis=0) / attr.sum()
        self.assertLess(share.max(), 0.6)

    def test_classical_detectors(self):
        for m in [IsolationForestAD(n_estimators=50), PCAReconstructionAD(n_components=0.8)]:
            m.fit(self.X_train_normal)
            m.calibrate_threshold(self.X_val_normal, percentile=90.0)
            s = m.predict_score(self.X_test)
            self.assertEqual(len(s), 50)
            self.assertGreater(s[40:].mean(), s[:40].mean())

    def test_supervised(self):
        y_train = np.r_[np.zeros(150), np.ones(150)].astype(int)
        X_train = np.vstack([self.X_train_normal[:150], self.X_train_normal[150:] + 2.0])
        for m in [RandomForestBaseline(n_estimators=30), GradientBoostingBaseline(n_estimators=30)]:
            m.fit(X_train, y_train)
            s = m.predict_score(X_train)
            self.assertGreater(roc_auc_score(y_train, s), 0.9)

    def test_column_rules(self):
        names = ["prev_finished_outcome", "hist_failure_rate_w5", "x"]
        X = np.array([[1, 0.2, 0], [0, 0.8, 0], [1, 1.0, 0]], dtype=float)
        prev = PreviousOutcomeBaseline(names).predict_score(X)
        rate = RollingFailureRateBaseline(names).predict_score(X)
        self.assertGreater(prev[0], prev[1])
        self.assertGreater(rate[1], rate[0])


class TestEvaluation(unittest.TestCase):
    def test_fast_metrics_match_sklearn(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 400)
        s = np.round(rng.random(400) + 0.4 * y, 2)
        self.assertAlmostEqual(fast_auroc(y, s), roc_auc_score(y, s), places=10)
        self.assertAlmostEqual(fast_auprc(y, s), average_precision_score(y, s), places=10)

    def test_block_bootstrap_detects_difference(self):
        rng = np.random.default_rng(2)
        strata = []
        for _ in range(3):
            y = rng.integers(0, 2, 300)
            good = y + rng.normal(0, 0.8, 300)
            bad = rng.normal(0, 1, 300)
            strata.append((y, good, bad))
        r = stratified_block_bootstrap(strata, metric="auroc", n_boot=300, block_length=10)
        self.assertGreater(r["observed_difference"], 0.1)
        self.assertLess(r["p_value"], 0.05)
        self.assertLess(r["ci_low"], r["ci_high"])

    def test_holm(self):
        c = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.20})
        self.assertTrue(c["a"]["reject_at_alpha"])
        self.assertFalse(c["b"]["reject_at_alpha"])
        self.assertEqual(c["a"]["family_size"], 3)

    def test_cost_models(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        pred = np.array([1, 0, 1, 0, 1, 0])
        job = np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        wall = job / 3
        c = cost_model_abort(y, pred, job, wall, fp_cost_minutes=2.0, tp_information_cost_minutes=15.0)
        self.assertAlmostEqual(c["compute_saved_min"], 120.0)
        self.assertAlmostEqual(c["net_saved_min"], 120.0 - 30.0 - 2.0)
        s = cost_model_skip(y, pred, job, missed_failure_cost_minutes=30.0)
        self.assertEqual(s["builds_skipped"], 3)
        self.assertEqual(s["failures_missed"], 1)
        self.assertAlmostEqual(s["net_saved_min"], 120.0 - 30.0)

    def test_rolling_threshold_uses_only_finished_builds(self):
        n = 50
        scores = np.linspace(0, 1, n)
        y = np.zeros(n, dtype=int)
        start = np.arange(n) * 100
        finish = start + 1000  # nothing finishes before the window ends
        warm = np.full(30, 0.5)
        preds = rolling_threshold_predictions(scores, y, start, finish, warm, percentile=92.0, window=200)
        # Threshold stays at the warm value 0.5 since no test build finishes in time.
        self.assertTrue((preds == (scores > 0.5)).all())


if __name__ == "__main__":
    unittest.main()
