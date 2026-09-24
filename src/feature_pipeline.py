#!/usr/bin/env python3
"""
feature_pipeline.py - Pre execution feature construction with an auditable
leakage boundary.

Every feature is a function of information that exists when a build starts:

  * attributes of the change and repository recorded at commit time,
  * the queue latency of the build itself, which is fixed before execution,
  * statistics of earlier builds that had FINISHED before this build started.

History is lagged by start order. On busy projects a large share of builds start while
the previous build is still running, so start order lagging exposed outcomes
that were not yet observable. History features are now computed over the set
of builds j with finish_j <= start_i, ordered by finish time.

Features are computed once over the whole project history and then split, so
the first builds of a validation or test window inherit the history of the
preceding window instead of an artificially truncated one.

The four feature groups used in the ablation are exposed as constants.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Columns that describe the outcome or are measured while the build runs.
POST_EXECUTION_COLUMNS = {
    "tr_status", "is_failure", "is_failure_strict", "tr_duration", "tr_purebuildduration",
    "tr_testduration", "tr_setup_time",
    "tr_tests_ok", "tr_tests_fail", "tr_tests_run", "tr_tests_skipped",
    "tr_tests_ran", "tr_tests_failed", "tr_failed_tests",
    "tr_job_count", "tr_num_jobs", "tr_jobs", "tr_analyzer", "tr_frameworks",
    "job_minutes_total", "n_jobs_passed", "n_jobs_failed", "n_jobs_errored",
    "n_jobs_canceled", "n_jobs_other", "latency_rows_disagree", "status_rows_disagree",
    "tr_finished_at",
}

# Counts that TravisTorrent collected at mining time over the whole life of
# the pull request or commit. They can include discussion that followed the
# build outcome, so they are treated as retrospective and excluded.
RETROSPECTIVE_COLUMNS = {
    "gh_num_issue_comments", "gh_num_pr_comments", "gh_num_commit_comments",
}

# Identifiers, free text and timestamps.
IDENTIFIER_COLUMNS = {
    "row", "tr_build_id", "tr_job_id", "tr_build_number", "tr_prev_build",
    "git_commit", "git_commits", "git_merged_with", "gh_pull_req_num",
    "gh_project_name", "gh_lang", "git_branch", "git_datetime",
    "tr_started_at", "gh_first_commit_created_at", "tr_lan",
}

HISTORY_FEATURES = [
    "prev_finished_outcome",
    "hist_failure_rate_w3", "hist_failure_rate_w5", "hist_failure_rate_w10",
    "hist_n_finished_w10",
]

TELEMETRY_FEATURES = [
    "hist_duration_mean_w3", "hist_duration_mean_w5", "hist_duration_mean_w10",
    "hist_duration_std_w3", "hist_duration_std_w5", "hist_duration_std_w10",
    "hist_latency_mean_w3", "hist_latency_mean_w5", "hist_latency_mean_w10",
    "duration_drift_w5", "current_ci_latency", "latency_surge_ratio",
    "time_since_prev_build_mins", "builds_running_at_start",
]

CHURN_FEATURES = [
    "git_diff_src_churn", "git_diff_test_churn", "churn_ratio", "churn_total",
    "git_num_all_built_commits", "git_num_committers", "gh_num_commits_on_files_touched",
    "commits_per_file", "gh_files_added", "gh_files_deleted", "gh_files_modified",
    "gh_tests_added", "gh_tests_deleted", "gh_src_files", "gh_doc_files", "gh_other_files",
    "gh_by_core_team_member", "is_pr_int", "gh_description_complexity",
]

REPOSITORY_FEATURES = [
    "gh_sloc", "log_sloc", "gh_team_size",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "history": HISTORY_FEATURES,
    "telemetry": TELEMETRY_FEATURES,
    "churn": CHURN_FEATURES,
    "repository": REPOSITORY_FEATURES,
}

# Named feature configurations for the ablation.
FEATURE_SUBSETS: Dict[str, List[str]] = {
    "all": ["history", "telemetry", "churn", "repository"],
    "history_only": ["history"],
    "telemetry_only": ["telemetry"],
    "churn_only": ["churn"],
    "repository_only": ["repository"],
    "no_repository": ["history", "telemetry", "churn"],
    "no_history": ["telemetry", "churn", "repository"],
}


def _finished_history(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Rolling statistics over builds that had finished before each start.

    Builds are ranked by finish time. For build i the number of finished
    builds is k_i = |{j : finish_j <= start_i}|; the window of size w is the
    last w builds in finish order among those k_i. Prefix sums make the
    statistics linear in the number of builds.
    """
    start = df["tr_started_at"].values.astype("datetime64[ns]").astype(np.int64)
    finish = df["tr_finished_at"].values.astype("datetime64[ns]").astype(np.int64)
    n = len(df)

    order = np.argsort(finish, kind="mergesort")
    finish_sorted = finish[order]
    outcome_sorted = df["is_failure"].values[order].astype(float)
    duration_sorted = df["tr_duration"].values[order].astype(float) / 60.0
    latency_sorted = np.nan_to_num(df["tr_ci_latency"].values[order].astype(float), nan=0.0)

    k = np.searchsorted(finish_sorted, start, side="right")

    def prefix(x):
        return np.concatenate([[0.0], np.cumsum(x)])

    p_out = prefix(outcome_sorted)
    p_dur = prefix(duration_sorted)
    p_dur2 = prefix(duration_sorted ** 2)
    p_lat = prefix(latency_sorted)

    out = {}
    for w in windows:
        lo = np.maximum(k - w, 0)
        cnt = (k - lo).astype(float)
        safe = np.maximum(cnt, 1.0)
        mean_out = (p_out[k] - p_out[lo]) / safe
        mean_dur = (p_dur[k] - p_dur[lo]) / safe
        var_dur = (p_dur2[k] - p_dur2[lo]) / safe - mean_dur ** 2
        std_dur = np.sqrt(np.maximum(var_dur, 0.0))
        mean_lat = (p_lat[k] - p_lat[lo]) / safe
        out[f"hist_failure_rate_w{w}"] = np.where(cnt > 0, mean_out, 0.0)
        out[f"hist_duration_mean_w{w}"] = np.where(cnt > 0, mean_dur, 0.0)
        out[f"hist_duration_std_w{w}"] = np.where(cnt > 1, std_dur, 0.0)
        out[f"hist_latency_mean_w{w}"] = np.where(cnt > 0, mean_lat, 0.0)
        out[f"hist_n_finished_w{w}"] = cnt

    out["prev_finished_outcome"] = np.where(k > 0, outcome_sorted[np.maximum(k - 1, 0)], 0.0)
    out["builds_running_at_start"] = (np.arange(n) - k).astype(float)
    return pd.DataFrame(out, index=df.index)


class TelemetryFeaturePipeline:
    def __init__(self, window_sizes: List[int] = [3, 5, 10], feature_subset: str = "all"):
        if feature_subset not in FEATURE_SUBSETS:
            raise ValueError(f"Unknown feature subset {feature_subset}")
        self.window_sizes = window_sizes
        self.feature_subset = feature_subset
        self.scaler = RobustScaler()
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all candidate features for a full, chronologically sorted project."""
        d = df.copy()

        def num(col, default=0.0):
            if col in d.columns:
                return pd.to_numeric(d[col], errors="coerce").fillna(default).astype(float)
            return pd.Series(default, index=d.index, dtype=float)

        src = num("git_diff_src_churn")
        tst = num("git_diff_test_churn")
        d["git_diff_src_churn"] = src
        d["git_diff_test_churn"] = tst
        d["churn_ratio"] = tst / (src + 1.0)
        d["churn_total"] = src + tst
        d["git_num_all_built_commits"] = num("git_num_all_built_commits")
        d["gh_num_commits_on_files_touched"] = num("gh_num_commits_on_files_touched")
        d["commits_per_file"] = d["git_num_all_built_commits"] / (d["gh_num_commits_on_files_touched"] + 1.0)
        d["gh_sloc"] = num("gh_sloc")
        d["log_sloc"] = np.log1p(d["gh_sloc"].clip(lower=0))
        d["is_pr_int"] = d["gh_is_pr"].astype(int) if "gh_is_pr" in d.columns else 0
        if "gh_by_core_team_member" in d.columns:
            d["gh_by_core_team_member"] = d["gh_by_core_team_member"].astype(str).str.lower().isin(["true", "1"]).astype(int)
        for col in ["gh_description_complexity", "gh_team_size", "git_num_committers",
                    "gh_files_added", "gh_files_deleted", "gh_files_modified",
                    "gh_tests_added", "gh_tests_deleted", "gh_src_files", "gh_doc_files",
                    "gh_other_files", "gh_test_lines_per_kloc", "gh_test_cases_per_kloc",
                    "gh_asserts_cases_per_kloc"]:
            d[col] = num(col)

        d["time_since_prev_build_mins"] = (
            d["tr_started_at"].diff().dt.total_seconds().fillna(3600.0) / 60.0
        ).clip(lower=0.1, upper=10080.0)

        # Queue latency of this build. tr_ci_latency is identical across the
        # job rows of a build in the dump (audited in the extraction manifest).
        d["current_ci_latency"] = num("tr_ci_latency").clip(lower=0.0)

        hist = _finished_history(d, self.window_sizes)
        for c in hist.columns:
            d[c] = hist[c].values

        d["duration_drift_w5"] = (
            (d["hist_duration_mean_w3"] - d["hist_duration_mean_w10"]) / (d["hist_duration_std_w10"] + 1.0)
        )
        d["latency_surge_ratio"] = d["current_ci_latency"] / (d["hist_latency_mean_w5"] + 1.0)
        return d

    def get_feature_names(self, d: pd.DataFrame) -> List[str]:
        exclude = POST_EXECUTION_COLUMNS | RETROSPECTIVE_COLUMNS | IDENTIFIER_COLUMNS | {
            "gh_is_pr", "tr_ci_latency",
            "gh_src_churn", "gh_test_churn", "git_num_commits", "gh_commits_on_files_touched",
        }
        groups = FEATURE_SUBSETS[self.feature_subset]
        wanted = [f for g in groups for f in FEATURE_GROUPS[g]]
        names = []
        for f in wanted:
            if f in exclude or f not in d.columns:
                continue
            if pd.api.types.is_numeric_dtype(d[f]) or pd.api.types.is_bool_dtype(d[f]):
                names.append(f)
        return names

    @staticmethod
    def _clean(block: pd.DataFrame) -> np.ndarray:
        return block.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-1e8, upper=1e8).values.astype(float)

    def fit(self, d_all: pd.DataFrame, train_idx: np.ndarray, fit_on_normal_only: bool = True):
        """Fit the scaler on the training rows (passing builds only by default)."""
        self.feature_columns = self.get_feature_names(d_all)
        block = d_all.iloc[train_idx]
        if fit_on_normal_only and (block["is_failure"] == 0).sum() >= 10:
            block = block[block["is_failure"] == 0]
        self.scaler.fit(self._clean(block[self.feature_columns]))
        self.is_fitted = True
        return self

    def transform_rows(self, d_all: pd.DataFrame, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        block = d_all.iloc[idx]
        X = self.scaler.transform(self._clean(block[self.feature_columns]))
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        X = np.clip(X, -50.0, 50.0)
        y = block["is_failure"].values.astype(int)
        return X, y

    # Convenience wrappers used by the tests.
    def fit_transform(self, train_df: pd.DataFrame, fit_on_normal_only: bool = True):
        d = self.transform_dataframe(train_df)
        self.fit(d, np.arange(len(d)), fit_on_normal_only=fit_on_normal_only)
        X, y = self.transform_rows(d, np.arange(len(d)))
        return X, y, self.feature_columns

    def transform(self, df: pd.DataFrame):
        d = self.transform_dataframe(df)
        return self.transform_rows(d, np.arange(len(d)))
