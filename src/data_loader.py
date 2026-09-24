#!/usr/bin/env python3
"""
data_loader.py - Ingestion and chronological splitting of build level CI data.

Reads the build level project files written by scripts/extract_projects.py,
derives start and finish timestamps, selects the label definition, and
produces chronological splits. Two split designs are supported:

  single_cut     one 70 / 10 / 20 chronological split, retained for
                 reproducibility and sensitivity analysis
  rolling_origin six expanding window folds; fold k trains on the first
                 30 + 10k percent of builds, calibrates on the next 10 percent
                 and tests on the 10 percent after that, so every build from
                 the 40 percent mark onwards is scored exactly once by a model
                 that has only seen earlier builds
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LABEL_LENIENT = "lenient"   # any job status other than passed
LABEL_STRICT = "strict"     # any job status failed or errored (cancelled excluded)


def load_build_dataset(filepath: str, label: str = LABEL_LENIENT) -> pd.DataFrame:
    """Load one build level project file and return it sorted by start time.

    The returned frame carries:
      tr_started_at   build start (pandas Timestamp)
      tr_finished_at  start plus the maximum job duration, the moment the
                      build outcome becomes observable
      is_failure      the selected label
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath, low_memory=False)

    col_mapping = {
        "gh_src_churn": "git_diff_src_churn",
        "gh_test_churn": "git_diff_test_churn",
        "git_num_commits": "git_num_all_built_commits",
        "gh_commits_on_files_touched": "gh_num_commits_on_files_touched",
    }
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    if "tr_started_at" not in df.columns and "git_datetime" in df.columns:
        df["tr_started_at"] = df["git_datetime"]
    df["tr_started_at"] = pd.to_datetime(df["tr_started_at"], errors="coerce")
    df = df.dropna(subset=["tr_started_at"]).copy()

    for c in ["tr_duration", "tr_ci_latency", "job_minutes_total",
              "git_diff_src_churn", "git_diff_test_churn",
              "git_num_all_built_commits", "gh_num_commits_on_files_touched", "gh_sloc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "tr_duration" not in df.columns:
        df["tr_duration"] = 0.0
    df["tr_duration"] = df["tr_duration"].fillna(0.0).clip(lower=0.0)
    if "job_minutes_total" not in df.columns:
        df["job_minutes_total"] = df["tr_duration"] / 60.0
    df["job_minutes_total"] = df["job_minutes_total"].fillna(df["tr_duration"] / 60.0)
    if "tr_ci_latency" not in df.columns:
        df["tr_ci_latency"] = 0.0
    df["tr_finished_at"] = df["tr_started_at"] + pd.to_timedelta(df["tr_duration"], unit="s")

    if label == LABEL_STRICT and "is_failure_strict" in df.columns:
        df["is_failure"] = pd.to_numeric(df["is_failure_strict"], errors="coerce").fillna(0).astype(int)
    elif "is_failure" in df.columns:
        df["is_failure"] = pd.to_numeric(df["is_failure"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_failure"] = (df["tr_status"].astype(str).str.lower() != "passed").astype(int)

    if "gh_is_pr" in df.columns:
        df["gh_is_pr"] = df["gh_is_pr"].astype(str).str.lower().isin(["true", "1", "yes"])

    if "tr_build_id" not in df.columns:
        df["tr_build_id"] = np.arange(len(df))
    df = df.sort_values(["tr_started_at", "tr_build_id"], kind="mergesort").reset_index(drop=True)
    df["git_datetime"] = df["tr_started_at"]
    return df


def concurrency_statistics(df: pd.DataFrame, lookback: int = 10) -> Dict[str, float]:
    """How often is the outcome of a recent build not yet observable at start?"""
    start = df["tr_started_at"].values.astype("datetime64[ns]").astype(np.int64)
    finish = df["tr_finished_at"].values.astype("datetime64[ns]").astype(np.int64)
    n = len(df)
    if n < 2:
        return {"prev_unfinished_pct": 0.0, "any_of_last10_unfinished_pct": 0.0,
                "mean_running_at_start": 0.0, "median_build_minutes": 0.0}
    prev_unfinished = finish[:-1] > start[1:]
    any_unfinished = np.zeros(n, dtype=bool)
    order = np.argsort(finish, kind="mergesort")
    finished_before = np.searchsorted(finish[order], start, side="right")
    running = np.arange(n) - finished_before
    for i in range(1, n):
        lo = max(0, i - lookback)
        any_unfinished[i] = (finish[lo:i] > start[i]).any()
    return {
        "prev_unfinished_pct": round(float(prev_unfinished.mean() * 100), 2),
        "any_of_last10_unfinished_pct": round(float(any_unfinished.mean() * 100), 2),
        "mean_running_at_start": round(float(running.mean()), 3),
        "median_build_minutes": round(float(np.median(df["tr_duration"].values) / 60.0), 2),
    }


def split_chronological(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.10):
    """Single chronological split, retained for reproducibility and sensitivity analysis."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def single_cut_fold(n: int, train_ratio: float = 0.70, val_ratio: float = 0.10) -> Dict[str, object]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {"fold": 0, "train_idx": np.arange(0, train_end),
            "val_idx": np.arange(train_end, val_end), "test_idx": np.arange(val_end, n)}


def rolling_origin_folds(n: int, first_train_frac: float = 0.30, step_frac: float = 0.10) -> List[Dict[str, object]]:
    """Expanding window folds; with the defaults there are six.

    Fold k trains on builds [0, 30 + 10k) percent, calibrates on
    [30 + 10k, 40 + 10k) and tests on [40 + 10k, 50 + 10k). The union of the
    six test windows is the last 60 percent of the project's history.
    """
    folds = []
    k = 0
    while True:
        train_end = int(round(n * (first_train_frac + k * step_frac)))
        val_end = int(round(n * (first_train_frac + (k + 1) * step_frac)))
        test_end = int(round(n * (first_train_frac + (k + 2) * step_frac)))
        if val_end >= n or test_end > n or test_end - val_end < 20:
            break
        folds.append({"fold": k, "train_idx": np.arange(0, train_end),
                      "val_idx": np.arange(train_end, val_end), "test_idx": np.arange(val_end, test_end)})
        k += 1
        if test_end == n:
            break
    return folds
