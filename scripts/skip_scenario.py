#!/usr/bin/env python3
"""Build skip scenario at a skip budget, for every project, fold and scorer.

Builds whose score falls in the lowest `skip_budget` share of the test window
are skipped. The saving is the compute (summed job minutes) of correctly
skipped passing builds; the cost is a swept penalty per skipped failing build,
representing failure observation delay. The scorers are refitted exactly as in
run_experiment.py (same folds, same seed, same hyperparameters); only the
operating point differs, so this script is a thin re evaluation rather than a
new experiment.

Output: results/skip_scenario.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from joblib import Parallel, delayed

from src.data_loader import LABEL_LENIENT, load_build_dataset, rolling_origin_folds
from src.feature_pipeline import TelemetryFeaturePipeline
from src.evaluation import cost_model_skip
from scripts.run_experiment import SCORERS, evaluate_fold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MISS_SWEEP = [30.0, 120.0, "one_build"]


def run_project(path: str, skip_budget: float, seed: int = 42):
    stem = os.path.splitext(os.path.basename(path))[0]
    df = load_build_dataset(path, label=LABEL_LENIENT)
    d = TelemetryFeaturePipeline().transform_dataframe(df)
    out = {"dataset": stem, "project": str(df["gh_project_name"].iloc[0]), "skip_budget": skip_budget, "folds": []}
    for fold in rolling_origin_folds(len(d)):
        _, _, (_, y_val), (_, y_te), scorers = evaluate_fold(d, fold, "all", SCORERS, seed=seed)
        rows = d.iloc[fold["test_idx"]]
        job_minutes = rows["job_minutes_total"].values.astype(float)
        wall = rows["tr_duration"].values.astype(float) / 60.0
        mean_wall = float(wall.mean())
        entry = {"fold": int(fold["fold"]), "n_test": int(len(y_te)), "prevalence_test": float(y_te.mean()),
                 "total_job_minutes": float(job_minutes.sum()), "scorers": {}}
        for name in SCORERS:
            te = scorers[name]["test"]
            n_run = max(1, int(round((1.0 - skip_budget) * len(te))))
            order = np.argsort(-te, kind="mergesort")
            pred = np.zeros(len(te), dtype=int)
            pred[order[:n_run]] = 1  # builds that are run; the rest are skipped
            entry["scorers"][name] = {}
            for c in MISS_SWEEP:
                miss = mean_wall if c == "one_build" else float(c)
                entry["scorers"][name][str(c)] = cost_model_skip(y_te, pred, job_minutes, missed_failure_cost_minutes=miss)
        out["folds"].append(entry)
    print(f"[{out['project']}] skip scenario done", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_budget", type=float, default=0.50)
    parser.add_argument("--n_jobs", type=int, default=7)
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, "data", "build_level", "*.csv")))
    results = Parallel(n_jobs=args.n_jobs)(delayed(run_project)(f, args.skip_budget) for f in files)
    payload = {"skip_budget": args.skip_budget, "projects": results}
    with open(os.path.join(ROOT, "results", "skip_scenario.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("Wrote results/skip_scenario.json")


if __name__ == "__main__":
    main()
