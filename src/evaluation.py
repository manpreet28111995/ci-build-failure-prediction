#!/usr/bin/env python3
"""
evaluation.py - Metrics, operating points, cost model and inference.

Contents
  fast_auroc / fast_auprc          rank based metrics used inside the bootstrap
  compute_metrics                  threshold free and operating point metrics
  matched_threshold                threshold giving a fixed alert budget on validation
  rolling_threshold_predictions    online percentile threshold over finished passing builds
  cost_model_abort                 abort or deprioritise predicted failures, with an
                                   information loss cost for true positives, a swept
                                   false alarm cost and compute credited as job minutes
  cost_model_skip                  skip predicted passing builds, with a missed failure cost
  stratified_block_bootstrap       paired moving block bootstrap on a metric difference,
                                   stratified by rolling origin fold
  holm_bonferroni                  Holm correction over a family of p values
  sign_test_above                  exact sign test for a pooled claim such as AUROC > 0.5
  mean_ci                          percentile bootstrap CI on a mean over projects
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.stats import rankdata


# ----------------------------------------------------------------------------
# Fast metrics
# ----------------------------------------------------------------------------

def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann Whitney AUROC with tie handling via average ranks."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (step wise area under the precision recall curve),
    identical to sklearn.metrics.average_precision_score up to tie handling."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    # Handle ties: use the precision at the last index of each tie group.
    distinct = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp_d = tp[distinct]
    prec_d = precision[distinct]
    tp_prev = np.r_[0, tp_d[:-1]]
    recall_step = (tp_d - tp_prev) / n_pos
    return float(np.sum(prec_d * recall_step))


def normalised_auprc(auprc: float, prevalence: float) -> float:
    """(AUPRC - p) / (1 - p): 0 for a random ranker, 1 for a perfect one."""
    if prevalence >= 1.0 or np.isnan(auprc):
        return float("nan")
    return float((auprc - prevalence) / (1.0 - prevalence))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    auroc = fast_auroc(y_true, y_scores)
    auprc = fast_auprc(y_true, y_scores)
    n_top = max(1, int(len(y_true) * 0.10))
    top_idx = np.argsort(-y_scores, kind="mergesort")[:n_top]
    alarm_rate = float(y_pred.mean()) if len(y_pred) else 0.0
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1), "fpr": float(fpr),
        "auroc": float(auroc), "auprc": float(auprc),
        "prevalence": prevalence,
        "auprc_lift": float(auprc / prevalence) if prevalence > 0 else float("nan"),
        "auprc_normalised": normalised_auprc(auprc, prevalence),
        "precision_at_10pct": float(np.mean(y_true[top_idx])),
        "alarm_rate": alarm_rate,
        "degenerate": bool(alarm_rate == 0.0 or alarm_rate == 1.0),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ----------------------------------------------------------------------------
# Operating points
# ----------------------------------------------------------------------------

def matched_threshold(val_scores: np.ndarray, alert_budget: float = 0.10) -> float:
    """Threshold such that a fraction alert_budget of validation builds is flagged."""
    return float(np.quantile(val_scores, 1.0 - alert_budget))


def rolling_threshold_predictions(
    scores: np.ndarray,
    y: np.ndarray,
    start_ns: np.ndarray,
    finish_ns: np.ndarray,
    warm_scores: np.ndarray,
    percentile: float = 92.0,
    window: int = 200,
) -> np.ndarray:
    """Online percentile threshold.

    For build i the threshold is the given percentile of the scores of the
    most recent `window` passing builds that had finished before start_i,
    drawn from the test window itself, seeded with the passing validation
    scores. Outcomes are used only once they are observable.
    """
    n = len(scores)
    preds = np.zeros(n, dtype=int)
    order = np.argsort(finish_ns, kind="mergesort")
    finish_sorted = finish_ns[order]
    pool = list(warm_scores[-window:])
    ptr = 0
    for i in range(n):
        while ptr < n and finish_sorted[ptr] <= start_ns[i]:
            j = order[ptr]
            if y[j] == 0:
                pool.append(scores[j])
                if len(pool) > window:
                    pool.pop(0)
            ptr += 1
        thr = np.percentile(pool, percentile) if len(pool) >= 20 else np.inf
        preds[i] = int(scores[i] > thr)
    return preds


# ----------------------------------------------------------------------------
# Cost models
# ----------------------------------------------------------------------------

def cost_model_abort(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    job_minutes: np.ndarray,
    wall_minutes: np.ndarray,
    fp_cost_minutes: float,
    tp_information_cost_minutes: float = 15.0,
) -> Dict[str, float]:
    """Intervention: a predicted failure is not run (or is deprioritised).

    Benefit  compute minutes of correctly withheld failing builds (sum of job
             minutes, which is what the build consumes)
    Cost     false alarm: fp_cost_minutes per passing build wrongly withheld
             true positive: tp_information_cost_minutes, the diagnostic value
             of the failing run that the developer no longer receives
    A missed failure (false negative) leaves the status quo unchanged and
    therefore carries no incremental cost in this scenario.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    compute_saved = float(job_minutes[tp_mask].sum())
    info_cost = float(tp_mask.sum() * tp_information_cost_minutes)
    fp_cost = float(fp_mask.sum() * fp_cost_minutes)
    net = compute_saved - info_cost - fp_cost
    total_failures = int(y_true.sum())
    # Break even false alarm cost: net = 0 solved for fp_cost.
    n_fp = int(fp_mask.sum())
    break_even = (compute_saved - info_cost) / n_fp if n_fp > 0 else float("inf")
    return {
        "compute_saved_min": round(compute_saved, 1),
        "information_cost_min": round(info_cost, 1),
        "false_alarm_cost_min": round(fp_cost, 1),
        "net_saved_min": round(net, 1),
        "net_saved_per_build_min": round(net / max(len(y_true), 1), 3),
        "mitigation_pct": round(tp_mask.sum() / total_failures * 100, 1) if total_failures else 0.0,
        "alarms": int(y_pred.sum()),
        "precision_at_operating_point": round(tp_mask.sum() / max(int(y_pred.sum()), 1), 3),
        "break_even_fp_cost_min": round(break_even, 1) if np.isfinite(break_even) else None,
        "mean_wall_minutes": round(float(wall_minutes.mean()), 1) if len(wall_minutes) else 0.0,
    }


def cost_model_skip(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    job_minutes: np.ndarray,
    missed_failure_cost_minutes: float,
) -> Dict[str, float]:
    """Intervention: builds predicted to pass are skipped (build selection).

    Benefit  compute minutes of correctly skipped passing builds
    Cost     missed_failure_cost_minutes per failing build that was skipped,
             representing delayed failure observation
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    skipped = y_pred == 0
    tn_mask = skipped & (y_true == 0)
    fn_mask = skipped & (y_true == 1)
    saved = float(job_minutes[tn_mask].sum())
    missed = int(fn_mask.sum())
    cost = missed * missed_failure_cost_minutes
    return {
        "builds_skipped": int(skipped.sum()),
        "skipped_pct": round(float(skipped.mean() * 100), 1),
        "compute_saved_min": round(saved, 1),
        "failures_missed": missed,
        "failures_missed_pct": round(missed / max(int(y_true.sum()), 1) * 100, 1),
        "missed_failure_cost_min": round(cost, 1),
        "net_saved_min": round(saved - cost, 1),
    }


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------

def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Moving block bootstrap indices of length n."""
    if block_length <= 1 or n <= block_length:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_length)[None, :]).ravel()[:n]
    return idx


def stratified_block_bootstrap(
    strata: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: str = "auroc",
    n_boot: int = 2000,
    block_length: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired moving block bootstrap of a metric difference, stratified by fold.

    strata is a list of (y, scores_a, scores_b) triples, one per rolling
    origin test window. The statistic is the mean over folds of
    metric(a) - metric(b). Within each fold, indices are resampled in moving
    blocks so that serial dependence in the build sequence is preserved;
    both detectors are evaluated on the same resampled indices.
    """
    fn = fast_auroc if metric == "auroc" else fast_auprc
    observed = []
    for y, a, b in strata:
        ma, mb = fn(y, a), fn(y, b)
        if not (np.isnan(ma) or np.isnan(mb)):
            observed.append(ma - mb)
    if not observed:
        return {"metric": metric, "observed_difference": None, "ci_low": None, "ci_high": None,
                "p_value": None, "n_bootstrap": 0, "block_length": block_length}
    observed_diff = float(np.mean(observed))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    diffs[:] = np.nan
    for r in range(n_boot):
        vals = []
        for y, a, b in strata:
            n = len(y)
            idx = _block_indices(n, block_length, rng)
            yb = y[idx]
            if yb.sum() == 0 or yb.sum() == n:
                continue
            d = fn(yb, a[idx]) - fn(yb, b[idx])
            if not np.isnan(d):
                vals.append(d)
        if vals:
            diffs[r] = np.mean(vals)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return {"metric": metric, "observed_difference": observed_diff, "ci_low": None, "ci_high": None,
                "p_value": None, "n_bootstrap": 0, "block_length": block_length}
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = float(min(1.0, 2.0 * tail))
    return {
        "metric": metric,
        "observed_difference": round(observed_diff, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "p_value": p_value,
        "p_value_floor": float(1.0 / diffs.size),
        "n_bootstrap": int(diffs.size),
        "block_length": block_length,
        "is_significant_at_05": bool(p_value < 0.05),
    }


def holm_bonferroni(p_values: Dict[str, Optional[float]], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm step down correction. Returns raw and adjusted p values and the
    rejection flag per comparison."""
    items = [(k, v) for k, v in p_values.items() if v is not None]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    corrected = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        running_max = max(running_max, adjusted)
        corrected[key] = {"raw_p": float(p), "adjusted_p": float(running_max),
                          "reject_at_alpha": bool(running_max < alpha), "family_size": m}
    return corrected


def sign_test_above(values: Sequence[float], threshold: float = 0.5) -> Dict[str, Any]:
    """Exact two sided sign test that the median exceeds threshold."""
    v = np.asarray([x for x in values if not np.isnan(x)])
    above = int((v > threshold).sum())
    below = int((v < threshold).sum())
    n = above + below
    if n == 0:
        return {"n": 0, "above": above, "p_value": None}
    res = stats.binomtest(above, n, 0.5, alternative="two-sided")
    return {"n": n, "above": above, "below": below, "p_value": float(res.pvalue)}


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5 or np.allclose(a, b):
        return {"n": int(len(a)), "p_value": None, "median_difference": float(np.median(a - b)) if len(a) else None}
    res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"n": int(len(a)), "p_value": float(res.pvalue), "median_difference": float(np.median(a - b))}


def mean_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for a mean over projects."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(v.mean()), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "median": round(float(np.median(v)), 4), "n": int(v.size)}


def format_p(p: Optional[float], floor: float = 0.0005) -> str:
    if p is None:
        return "n/a"
    if p < floor:
        return f"< {floor:g}"
    return f"{p:.3f}"
