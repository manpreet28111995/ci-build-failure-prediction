#!/usr/bin/env python3
"""Invariant tests for loading, completion time lagging and the leakage boundary."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import concurrency_statistics, load_build_dataset, rolling_origin_folds  # noqa: E402
from src.feature_pipeline import (POST_EXECUTION_COLUMNS, RETROSPECTIVE_COLUMNS,  # noqa: E402
                                  TelemetryFeaturePipeline)


def make_fixture(path, n=120, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.date_range("2016-01-01", periods=n, freq="30min")
    duration = rng.integers(600, 5400, n)  # 10 to 90 minutes, so builds overlap
    df = pd.DataFrame({
        "tr_build_id": np.arange(1, n + 1),
        "gh_project_name": ["test/repo"] * n,
        "gh_lang": ["ruby"] * n,
        "git_branch": ["master"] * n,
        "gh_is_pr": [False] * n,
        "tr_started_at": start.strftime("%Y-%m-%d %H:%M:%S"),
        "tr_duration": duration,
        "job_minutes_total": duration / 60.0 * 3,
        "tr_ci_latency": rng.uniform(5, 30, n),
        "gh_sloc": np.linspace(10000, 12000, n),
        "gh_team_size": [10] * n,
        "gh_src_churn": rng.integers(10, 500, n),
        "gh_test_churn": rng.integers(0, 200, n),
        "git_num_commits": rng.integers(1, 5, n),
        "gh_commits_on_files_touched": rng.integers(1, 10, n),
        "gh_num_pr_comments": rng.integers(0, 5, n),
        "tr_status": ["passed"] * 90 + ["failed"] * 30,
        "is_failure": [0] * 90 + [1] * 30,
        "tr_tests_fail": rng.integers(0, 3, n),
    })
    df.to_csv(path, index=False)
    return df


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(os.path.dirname(__file__), "_fixture.csv")
        make_fixture(cls.path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.path):
            os.remove(cls.path)

    def test_loader_sorts_and_derives_finish(self):
        df = load_build_dataset(self.path)
        self.assertTrue(df["tr_started_at"].is_monotonic_increasing)
        self.assertTrue((df["tr_finished_at"] >= df["tr_started_at"]).all())
        stats = concurrency_statistics(df)
        self.assertGreater(stats["prev_unfinished_pct"], 0.0)

    def test_history_uses_only_finished_builds(self):
        df = load_build_dataset(self.path)
        d = TelemetryFeaturePipeline().transform_dataframe(df)
        for i in [10, 50, 100]:
            start = d.loc[i, "tr_started_at"]
            finished = d[d["tr_finished_at"] <= start].sort_values("tr_finished_at")
            self.assertEqual(min(len(finished), 10), d.loc[i, "hist_n_finished_w10"])
            if len(finished):
                self.assertEqual(d.loc[i, "prev_finished_outcome"], finished.iloc[-1]["is_failure"])
                self.assertAlmostEqual(d.loc[i, "hist_failure_rate_w5"], finished.tail(5)["is_failure"].mean())
            self.assertEqual(d.loc[i, "builds_running_at_start"], i - len(finished))

    def test_leakage_boundary(self):
        df = load_build_dataset(self.path)
        d = TelemetryFeaturePipeline().transform_dataframe(df)
        pipe = TelemetryFeaturePipeline()
        pipe.fit(d, np.arange(80))
        names = set(pipe.feature_columns)
        self.assertFalse(names & POST_EXECUTION_COLUMNS)
        self.assertFalse(names & RETROSPECTIVE_COLUMNS)
        self.assertNotIn("tr_tests_fail", names)
        X, y = pipe.transform_rows(d, np.arange(80, 120))
        self.assertEqual(X.shape[0], 40)
        self.assertFalse(np.isnan(X).any())

    def test_rolling_origin_folds_are_forward_only(self):
        folds = rolling_origin_folds(1000)
        self.assertEqual(len(folds), 6)
        for f in folds:
            self.assertLess(f["train_idx"].max(), f["val_idx"].min())
            self.assertLess(f["val_idx"].max(), f["test_idx"].min())
        self.assertEqual(folds[-1]["test_idx"][-1], 999)

    def test_feature_subsets(self):
        df = load_build_dataset(self.path)
        d = TelemetryFeaturePipeline().transform_dataframe(df)
        counts = {}
        for subset in ["all", "history_only", "telemetry_only", "churn_only", "repository_only"]:
            p = TelemetryFeaturePipeline(feature_subset=subset)
            p.fit(d, np.arange(80))
            counts[subset] = len(p.feature_columns)
        self.assertEqual(counts["all"], counts["history_only"] + counts["telemetry_only"] + counts["churn_only"] + counts["repository_only"])


if __name__ == "__main__":
    unittest.main()
