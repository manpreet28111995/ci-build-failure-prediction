#!/usr/bin/env python3
"""
run_experiment.py - Benchmark runner for leakage explicit build outcome prediction.

For every build level project file the runner:

  1. computes the pre execution feature frame once (completion time lagged);
  2. evaluates eight scorers under rolling origin evaluation (six expanding
     window folds) and, for comparison, under the single 70 / 10 / 20 cut:
        PREV     outcome of the last finished build
        LAST5    mean outcome of the last five finished builds
        HIST-LR  logistic regression on the history group only
        AE       reconstruction autoencoder (repaired variance weighting)
        IF       isolation forest
        PCA      PCA reconstruction
        RF       random forest
        GB       histogram gradient boosting
  3. reports threshold free metrics, and operating point metrics at
        native thresholds (92nd percentile of passing validation scores for
        the unsupervised detectors, 0.5 for the others, as in the first
        submission),
        a matched alert budget (the top ten percent of validation scores for
        every scorer), and
        a rolling threshold for the unsupervised detectors;
  4. evaluates two cost models at the matched operating point with a sweep
     over the false alarm cost and the missed failure cost;
  5. runs a four group ablation (history / telemetry / churn / repository) on
     every learned detector;
  6. compares every scorer against two references (LAST5 and RF) with a
     paired moving block bootstrap stratified by fold, with Holm correction
     within project, and reports block length sensitivity;
  7. reruns the main configuration under the strict label (failed or errored
     jobs only) as a label sensitivity check;
  8. stores the fold scores of the final fold so figures can be regenerated.

Outputs: results/<project>_results.json and results/<project>_scores.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List

# Single threaded numerical libraries: the projects are parallelised with
# joblib, and OpenMP oversubscription slows histogram gradient boosting by an
# order of magnitude on small tables.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data_loader import (LABEL_LENIENT, LABEL_STRICT, concurrency_statistics,
                             load_build_dataset, rolling_origin_folds, single_cut_fold)
from src.feature_pipeline import FEATURE_GROUPS, HISTORY_FEATURES, TelemetryFeaturePipeline
from src.models.autoencoder import TelemetryAutoencoder
from src.models.classical_ad import IsolationForestAD, PCAReconstructionAD
from src.models.baselines import (GradientBoostingBaseline, HistoryLogisticRegression,
                                  PreviousOutcomeBaseline, RandomForestBaseline,
                                  RollingFailureRateBaseline)
from src.evaluation import (compute_metrics, cost_model_abort, cost_model_skip, holm_bonferroni,
                            matched_threshold, rolling_threshold_predictions,
                            stratified_block_bootstrap)

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCORERS = ["PREV", "LAST5", "HIST-LR", "AE", "IF", "PCA", "RF", "GB"]
UNSUPERVISED = {"AE", "IF", "PCA"}
LEARNED = ["AE", "IF", "PCA", "RF", "GB"]
ABLATION_SUBSETS = ["history_only", "telemetry_only", "churn_only", "repository_only", "no_repository", "no_history"]
FP_COST_SWEEP = [2.0, 5.0, 15.0, 30.0, "one_build"]
MISSED_FAILURE_COST_SWEEP = [30.0, 120.0, "one_build"]
ALERT_BUDGET = 0.10
NATIVE_PERCENTILE = 92.0
TP_INFORMATION_COST = 15.0
BLOCK_LENGTH = 10
BLOCK_SENSITIVITY = [1, 5, 10, 20, 40]


def fit_scorers(X_tr, y_tr, X_val, y_val, X_te, feature_names, which: List[str], seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit the requested scorers and return validation and test scores."""
    out = {}
    X_tr_normal = X_tr[y_tr == 0] if (y_tr == 0).sum() >= 20 else X_tr
    X_val_normal = X_val[y_val == 0] if (y_val == 0).sum() >= 10 else X_val
    for name in which:
        if name == "PREV":
            m = PreviousOutcomeBaseline(feature_names)
        elif name == "LAST5":
            m = RollingFailureRateBaseline(feature_names, window=5)
        elif name == "HIST-LR":
            m = HistoryLogisticRegression(feature_names, HISTORY_FEATURES, random_state=seed)
        elif name == "AE":
            m = TelemetryAutoencoder(random_state=seed)
        elif name == "IF":
            m = IsolationForestAD(random_state=seed)
        elif name == "PCA":
            m = PCAReconstructionAD(random_state=seed)
        elif name == "RF":
            m = RandomForestBaseline(random_state=seed)
        elif name == "GB":
            m = GradientBoostingBaseline(random_state=seed)
        else:
            raise ValueError(name)
        if name in UNSUPERVISED:
            m.fit(X_tr_normal)
            native_thr = m.calibrate_threshold(X_val_normal, percentile=NATIVE_PERCENTILE)
        else:
            m.fit(X_tr, y_tr)
            native_thr = 0.5
        out[name] = {
            "val": m.predict_score(X_val),
            "test": m.predict_score(X_te),
            "native_threshold": float(native_thr),
            "model": m,
        }
    return out


def evaluate_fold(d, fold, feature_subset: str, which: List[str], seed: int = 42):
    pipe = TelemetryFeaturePipeline(window_sizes=[3, 5, 10], feature_subset=feature_subset)
    pipe.fit(d, fold["train_idx"], fit_on_normal_only=True)
    X_tr, y_tr = pipe.transform_rows(d, fold["train_idx"])
    X_val, y_val = pipe.transform_rows(d, fold["val_idx"])
    X_te, y_te = pipe.transform_rows(d, fold["test_idx"])
    scorers = fit_scorers(X_tr, y_tr, X_val, y_val, X_te, pipe.feature_columns, which, seed=seed)
    return pipe, (X_tr, y_tr), (X_val, y_val), (X_te, y_te), scorers


def summarise_operating_points(d, fold, y_val, y_te, scorers) -> Dict[str, Any]:
    """Metrics at native, matched and rolling operating points, plus cost models."""
    test_rows = d.iloc[fold["test_idx"]]
    job_minutes = test_rows["job_minutes_total"].values.astype(float)
    wall_minutes = test_rows["tr_duration"].values.astype(float) / 60.0
    start_ns = test_rows["tr_started_at"].values.astype("datetime64[ns]").astype(np.int64)
    finish_ns = test_rows["tr_finished_at"].values.astype("datetime64[ns]").astype(np.int64)
    mean_wall = float(wall_minutes.mean()) if len(wall_minutes) else 0.0

    result = {}
    for name, s in scorers.items():
        te, val = s["test"], s["val"]
        native_pred = (te > s["native_threshold"]).astype(int) if name in UNSUPERVISED else (te >= s["native_threshold"]).astype(int)
        thr_matched = matched_threshold(val, ALERT_BUDGET)
        matched_val_pred = (te > thr_matched).astype(int)
        # Equal alert budget by rank: exactly the top ALERT_BUDGET share of the
        # test window is flagged for every scorer, so operating point
        # comparisons differ only in which builds each scorer ranks highest.
        n_alarms = max(1, int(round(ALERT_BUDGET * len(te))))
        rank_order = np.argsort(-te, kind="mergesort")
        matched_pred = np.zeros(len(te), dtype=int)
        matched_pred[rank_order[:n_alarms]] = 1
        entry = {
            "threshold_free": compute_metrics(y_te, matched_pred, te),
            "native": compute_metrics(y_te, native_pred, te),
            "matched_validation": compute_metrics(y_te, matched_val_pred, te),
            "matched": compute_metrics(y_te, matched_pred, te),
        }
        if name in UNSUPERVISED:
            warm = val[y_val == 0]
            rolling_pred = rolling_threshold_predictions(te, y_te, start_ns, finish_ns, warm,
                                                         percentile=NATIVE_PERCENTILE, window=200)
            entry["rolling"] = compute_metrics(y_te, rolling_pred, te)
        cost_abort = {}
        for c in FP_COST_SWEEP:
            fp_cost = mean_wall if c == "one_build" else float(c)
            cost_abort[str(c)] = cost_model_abort(y_te, matched_pred, job_minutes, wall_minutes,
                                                  fp_cost_minutes=fp_cost,
                                                  tp_information_cost_minutes=TP_INFORMATION_COST)
        cost_skip = {}
        for c in MISSED_FAILURE_COST_SWEEP:
            miss_cost = mean_wall if c == "one_build" else float(c)
            cost_skip[str(c)] = cost_model_skip(y_te, matched_pred, job_minutes, missed_failure_cost_minutes=miss_cost)
        entry["cost_abort_matched"] = cost_abort
        entry["cost_skip_matched"] = cost_skip
        result[name] = entry
    result["_context"] = {
        "mean_wall_minutes": round(mean_wall, 2),
        "mean_job_minutes": round(float(job_minutes.mean()), 2) if len(job_minutes) else 0.0,
        "total_job_minutes": round(float(job_minutes.sum()), 1),
        "failing_job_minutes": round(float(job_minutes[y_te == 1].sum()), 1),
    }
    return result


def run_protocol(d: pd.DataFrame, folds: List[Dict[str, Any]], with_ablation: bool, seed: int = 42):
    per_fold = []
    strata_scores = {name: [] for name in SCORERS}
    strata_y = []
    final_fold_payload = None
    for fold in folds:
        t0 = time.time()
        pipe, (X_tr, y_tr), (X_val, y_val), (X_te, y_te), scorers = evaluate_fold(d, fold, "all", SCORERS, seed=seed)
        ops = summarise_operating_points(d, fold, y_val, y_te, scorers)
        fold_entry = {
            "fold": int(fold["fold"]),
            "n_train": int(len(fold["train_idx"])), "n_val": int(len(fold["val_idx"])), "n_test": int(len(fold["test_idx"])),
            "prevalence_train": round(float(y_tr.mean()), 4),
            "prevalence_val": round(float(y_val.mean()), 4),
            "prevalence_test": round(float(y_te.mean()), 4),
            "test_failures": int(y_te.sum()),
            "test_start": str(d.iloc[fold["test_idx"][0]]["tr_started_at"]),
            "test_end": str(d.iloc[fold["test_idx"][-1]]["tr_started_at"]),
            "feature_count": len(pipe.feature_columns),
            "scorers": {k: v for k, v in ops.items() if k != "_context"},
            "context": ops["_context"],
        }
        # Autoencoder diagnostics: share of the score carried by the largest feature.
        ae = scorers["AE"]["model"]
        attr = ae.compute_feature_attribution(X_te)
        share = attr.sum(axis=0) / max(attr.sum(), 1e-12)
        top = int(np.argmax(share))
        fold_entry["ae_diagnostics"] = {
            "top_feature": pipe.feature_columns[top],
            "top_feature_share_pct": round(float(share[top] * 100), 1),
            "active_features": int(ae.active_features.sum()),
            "native_threshold": scorers["AE"]["native_threshold"],
            "test_median_score": round(float(np.median(scorers["AE"]["test"])), 4),
            "test_alarm_rate_native_pct": round(float((scorers["AE"]["test"] > scorers["AE"]["native_threshold"]).mean() * 100), 1),
        }
        if with_ablation:
            ab = {}
            for subset in ABLATION_SUBSETS:
                which = LEARNED + (["PREV", "LAST5", "HIST-LR"] if subset == "history_only" else [])
                _, _, (_, yv2), (_, yt2), sc2 = evaluate_fold(d, fold, subset, which, seed=seed)
                ab[subset] = {}
                for name, s in sc2.items():
                    m = compute_metrics(yt2, (s["test"] >= np.quantile(s["test"], 1 - ALERT_BUDGET)).astype(int), s["test"])
                    ab[subset][name] = {"auroc": m["auroc"], "auprc": m["auprc"], "auprc_lift": m["auprc_lift"],
                                        "auprc_normalised": m["auprc_normalised"]}
            fold_entry["ablation"] = ab
        fold_entry["seconds"] = round(time.time() - t0, 1)
        per_fold.append(fold_entry)
        strata_y.append(y_te)
        for name in SCORERS:
            strata_scores[name].append(scorers[name]["test"])
        final_fold_payload = {
            "y_test": y_te,
            "feature_names": pipe.feature_columns,
            "scores": {name: scorers[name]["test"] for name in SCORERS},
            "val_scores": {name: scorers[name]["val"] for name in SCORERS},
            "y_val": y_val,
            "ae_attribution_pass": attr[y_te == 0].mean(axis=0),
            "ae_attribution_fail": attr[y_te == 1].mean(axis=0) if (y_te == 1).any() else np.zeros(attr.shape[1]),
            "ae_native_threshold": scorers["AE"]["native_threshold"],
            "X_test": X_te,
            "X_train_normal": X_tr[y_tr == 0],
        }
    return per_fold, strata_y, strata_scores, final_fold_payload


def legacy_autoencoder_diagnostics(payload, seed: int = 42) -> Dict[str, Any]:
    """Reproduce legacy inverse-variance weighting for diagnostic comparison."""
    X_tr = payload["X_train_normal"]
    X_te = payload["X_test"]
    ae = TelemetryAutoencoder(random_state=seed)
    ae.fit(X_tr)
    var = np.var(X_tr, axis=0)
    legacy_w = 1.0 / (var + 1e-4)
    diff_sq = (X_te - ae.reconstruct(X_te)) ** 2
    legacy_attr = diff_sq * legacy_w
    share = legacy_attr.sum(axis=0) / max(legacy_attr.sum(), 1e-12)
    top = int(np.argmax(share))
    legacy_scores = legacy_attr.mean(axis=1)
    y = payload["y_test"]
    m = compute_metrics(y, (legacy_scores > np.inf).astype(int), legacy_scores)
    return {
        "top_feature": payload["feature_names"][top],
        "top_feature_share_pct": round(float(share[top] * 100), 1),
        "zero_variance_features": int((var < 1e-8).sum()),
        "auroc": m["auroc"], "auprc": m["auprc"],
        "scores": legacy_scores,
    }


def run_project(dataset_path: str, output_dir: str, n_boot: int, seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    t_start = time.time()
    df = load_build_dataset(dataset_path, label=LABEL_LENIENT)
    pipe0 = TelemetryFeaturePipeline()
    d = pipe0.transform_dataframe(df)
    n = len(d)
    project = str(df["gh_project_name"].iloc[0]) if "gh_project_name" in df.columns else stem
    print(f"[{project}] {n} builds, prevalence {df['is_failure'].mean():.3f}", flush=True)

    conc = concurrency_statistics(df)
    folds = rolling_origin_folds(n)
    single = [single_cut_fold(n)]

    rolling_folds, strata_y, strata_scores, final_payload = run_protocol(d, folds, with_ablation=not quick, seed=seed)
    single_folds, _, _, _ = run_protocol(d, single, with_ablation=False, seed=seed)

    # Inference: paired stratified block bootstrap against two references.
    comparisons = {}
    for ref in ["LAST5", "RF"]:
        comparisons[ref] = {}
        for metric in ["auroc", "auprc"]:
            tests = {}
            for name in SCORERS:
                if name == ref:
                    continue
                strata = [(strata_y[i], strata_scores[name][i], strata_scores[ref][i]) for i in range(len(strata_y))]
                tests[name] = stratified_block_bootstrap(strata, metric=metric, n_boot=n_boot,
                                                         block_length=BLOCK_LENGTH, seed=seed)
            holm = holm_bonferroni({k: v["p_value"] for k, v in tests.items()})
            for k in tests:
                tests[k]["holm"] = holm.get(k)
            comparisons[ref][metric] = tests

    sensitivity = {}
    for label, a, b in [("AE_vs_RF", "AE", "RF"), ("GB_vs_LAST5", "GB", "LAST5"), ("RF_vs_LAST5", "RF", "LAST5")]:
        sensitivity[label] = {}
        for L in BLOCK_SENSITIVITY:
            strata = [(strata_y[i], strata_scores[a][i], strata_scores[b][i]) for i in range(len(strata_y))]
            r = stratified_block_bootstrap(strata, metric="auroc", n_boot=max(500, n_boot // 2), block_length=L, seed=seed)
            sensitivity[label][str(L)] = {"ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"],
                                          "observed_difference": r["observed_difference"]}

    # Label sensitivity: strict label, rolling origin, no ablation, no bootstrap.
    strict = {}
    if not quick:
        df_s = load_build_dataset(dataset_path, label=LABEL_STRICT)
        if df_s["is_failure"].sum() > 0 and df_s["is_failure"].mean() != df["is_failure"].mean():
            d_s = pipe0.transform_dataframe(df_s)
            strict_folds, _, _, _ = run_protocol(d_s, rolling_origin_folds(len(d_s)), with_ablation=False, seed=seed)
            strict = {"prevalence": round(float(df_s["is_failure"].mean()), 4),
                      "builds_relabelled": int((df_s["is_failure"] != df["is_failure"]).sum()),
                      "folds": [{"fold": f["fold"], "prevalence_test": f["prevalence_test"],
                                 "scorers": {k: {"auroc": v["threshold_free"]["auroc"], "auprc": v["threshold_free"]["auprc"],
                                                 "auprc_lift": v["threshold_free"]["auprc_lift"]}
                                             for k, v in f["scorers"].items()}} for f in strict_folds]}
        else:
            strict = {"prevalence": round(float(df_s["is_failure"].mean()), 4), "builds_relabelled": 0, "folds": []}

    legacy = legacy_autoencoder_diagnostics(final_payload, seed=seed)

    summary = {
        "dataset": stem,
        "project": project,
        "language": str(df["gh_lang"].iloc[0]) if "gh_lang" in df.columns else "unknown",
        "first_build": str(df["tr_started_at"].min()),
        "last_build": str(df["tr_started_at"].max()),
        "total_builds": int(n),
        "total_failures": int(df["is_failure"].sum()),
        "failure_rate_pct": round(float(df["is_failure"].mean() * 100), 2),
        "mean_jobs_per_build": round(float(df["tr_job_count"].mean()), 2) if "tr_job_count" in df.columns else None,
        "median_wall_minutes": round(float(df["tr_duration"].median() / 60.0), 1),
        "median_job_minutes_total": round(float(df["job_minutes_total"].median()), 1),
        "latency_rows_disagree_pct": round(float(df["latency_rows_disagree"].mean() * 100), 3) if "latency_rows_disagree" in df.columns else None,
        "status_rows_disagree_pct": round(float(df["status_rows_disagree"].mean() * 100), 3) if "status_rows_disagree" in df.columns else None,
        "concurrency": conc,
        "feature_names": final_payload["feature_names"],
        "feature_count": len(final_payload["feature_names"]),
        "feature_groups": {g: [f for f in fs if f in final_payload["feature_names"]] for g, fs in FEATURE_GROUPS.items()},
        "protocol": {
            "rolling_origin": {"n_folds": len(folds), "alert_budget": ALERT_BUDGET, "native_percentile": NATIVE_PERCENTILE,
                               "tp_information_cost_min": TP_INFORMATION_COST, "block_length": BLOCK_LENGTH, "n_bootstrap": n_boot},
        },
        "rolling_origin": rolling_folds,
        "single_cut": single_folds,
        "comparisons": comparisons,
        "block_length_sensitivity": sensitivity,
        "label_sensitivity_strict": strict,
        "legacy_autoencoder": {k: v for k, v in legacy.items() if k != "scores"},
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{stem}_results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    np.savez_compressed(
        os.path.join(output_dir, f"{stem}_scores.npz"),
        y_test=final_payload["y_test"],
        feature_names=np.array(final_payload["feature_names"]),
        ae_attribution_pass=final_payload["ae_attribution_pass"],
        ae_attribution_fail=final_payload["ae_attribution_fail"],
        ae_native_threshold=np.array([final_payload["ae_native_threshold"]]),
        legacy_ae_scores=legacy["scores"],
        **{f"score_{k}": v for k, v in final_payload["scores"].items()},
        **{f"val_{k}": v for k, v in final_payload["val_scores"].items()},
        y_val=final_payload["y_val"],
    )
    print(f"[{project}] done in {summary['runtime_seconds']} s", flush=True)
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(str(type(o)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the leakage explicit build prediction benchmark.")
    parser.add_argument("--data_dir", default=os.path.join(ROOT, "data", "build_level"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--projects", nargs="*", default=None, help="dataset file stems to run (default: all)")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_jobs", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="skip ablation and label sensitivity")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if args.projects:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in set(args.projects)]
    if not files:
        raise SystemExit("No build level CSV files found. Run scripts/extract_projects.py first.")
    print(f"Running {len(files)} projects with n_boot={args.n_boot}, n_jobs={args.n_jobs}")
    Parallel(n_jobs=args.n_jobs)(delayed(run_project)(f, args.output_dir, args.n_boot, quick=args.quick) for f in files)
    print("All projects finished.")


if __name__ == "__main__":
    main()
