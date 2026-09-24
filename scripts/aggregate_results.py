#!/usr/bin/env python3
"""Aggregate per project result files into cross project tables.

Reads results/<project>_results.json and writes
  results/cross_project_summary.json
  docs/*.tex   one LaTeX table per file, included by the paper
  docs/summary.md a Markdown digest of the same numbers

The primary sample is the ten projects marked "primary" in the extraction
manifest; the four "robustness" projects are pooled separately.
Every number in the paper's tables comes from these files.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import format_p, mean_ci, sign_test_above, wilcoxon_paired  # noqa: E402

SCORERS = ["PREV", "LAST5", "HIST-LR", "AE", "IF", "PCA", "RF", "GB"]
LEARNED = ["AE", "IF", "PCA", "RF", "GB"]
LABEL = {"PREV": "Previous outcome", "LAST5": "Last five rate", "HIST-LR": "History LR",
         "AE": "Autoencoder", "IF": "Isolation forest", "PCA": "PCA reconstruction",
         "RF": "Random forest", "GB": "Gradient boosting"}
ABLATION_SUBSETS = ["history_only", "telemetry_only", "churn_only", "repository_only", "no_repository", "no_history", "all"]
ABLATION_LABEL = {"history_only": "History only", "telemetry_only": "Telemetry only", "churn_only": "Churn only",
                  "repository_only": "Repository only", "no_repository": "All minus repository",
                  "no_history": "All minus history", "all": "All four groups"}
FP_SWEEP = ["2.0", "5.0", "15.0", "30.0", "one_build"]
FP_LABEL = {"2.0": "2 min", "5.0": "5 min", "15.0": "15 min", "30.0": "30 min", "one_build": "one build"}
MISS_SWEEP = ["30.0", "120.0", "one_build"]
MISS_LABEL = {"30.0": "30 min", "120.0": "120 min", "one_build": "one build"}


def esc(text: str) -> str:
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def short_project(name: str) -> str:
    return name.split("/")[-1]


def fmt(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s


def load_runs(results_dir: Path):
    runs = []
    for f in sorted(glob.glob(str(results_dir / "*_results.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            runs.append(json.load(fh))
    manifest = json.loads((ROOT / "data" / "build_level" / "EXTRACTION_MANIFEST.json").read_text(encoding="utf-8"))["projects"]
    ranking = json.loads((ROOT / "data" / "build_level" / "PROJECT_RANKING.json").read_text(encoding="utf-8"))
    rank_of = {r["project"]: r["rank"] for r in ranking["top_30_by_distinct_builds"]}
    for r in runs:
        m = manifest.get(r["project"], {})
        r["role"] = m.get("role", "primary")
        r["manifest"] = m
        r["rank"] = rank_of.get(r["project"])
    runs.sort(key=lambda r: (0 if r["role"] == "primary" else 1, -r["total_builds"]))
    return runs


def fold_mean(run, scorer, key, protocol="rolling_origin", block="threshold_free"):
    vals = [f["scorers"][scorer][block][key] for f in run[protocol] if f["scorers"][scorer][block].get(key) is not None]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def per_project_table(runs):
    rows = []
    for r in runs:
        row = {"project": r["project"], "role": r["role"], "rank": r["rank"], "language": r["language"],
               "builds": r["total_builds"], "failure_rate_pct": r["failure_rate_pct"],
               "strict_failure_rate_pct": r["manifest"].get("failure_rate_strict_pct"),
               "mean_jobs_per_build": r["mean_jobs_per_build"], "median_wall_minutes": r["median_wall_minutes"],
               "median_job_minutes_total": r["median_job_minutes_total"],
               "concurrency": r["concurrency"], "feature_count": r["feature_count"],
               "test_prevalence_mean": float(np.mean([f["prevalence_test"] for f in r["rolling_origin"]])),
               "fold_prevalence": [f["prevalence_test"] for f in r["rolling_origin"]],
               "fold_train_prevalence": [f["prevalence_train"] for f in r["rolling_origin"]],
               "fold_failures": [f["test_failures"] for f in r["rolling_origin"]],
               "n_folds": len(r["rolling_origin"]),
               "metrics": {}}
        for s in SCORERS:
            row["metrics"][s] = {
                "auroc": fold_mean(r, s, "auroc"),
                "auprc": fold_mean(r, s, "auprc"),
                "auprc_lift": fold_mean(r, s, "auprc_lift"),
                "auprc_normalised": fold_mean(r, s, "auprc_normalised"),
                "fold_auroc": [f["scorers"][s]["threshold_free"]["auroc"] for f in r["rolling_origin"]],
                "matched_precision": fold_mean(r, s, "precision", block="matched"),
                "matched_recall": fold_mean(r, s, "recall", block="matched"),
                "matched_alarm_rate": fold_mean(r, s, "alarm_rate", block="matched"),
                "native_alarm_rate": fold_mean(r, s, "alarm_rate", block="native"),
                "native_precision": fold_mean(r, s, "precision", block="native"),
                "native_recall": fold_mean(r, s, "recall", block="native"),
                "native_degenerate_folds": int(sum(1 for f in r["rolling_origin"] if f["scorers"][s]["native"]["degenerate"])),
                "single_cut_auroc": r["single_cut"][0]["scorers"][s]["threshold_free"]["auroc"],
                "single_cut_lift": r["single_cut"][0]["scorers"][s]["threshold_free"]["auprc_lift"],
            }
            if "rolling" in r["rolling_origin"][0]["scorers"][s]:
                row["metrics"][s]["rolling_alarm_rate"] = fold_mean(r, s, "alarm_rate", block="rolling")
                row["metrics"][s]["rolling_precision"] = fold_mean(r, s, "precision", block="rolling")
                row["metrics"][s]["rolling_recall"] = fold_mean(r, s, "recall", block="rolling")
            # Cost models, summed over folds (i.e. over the last 60 percent of history).
            abort = {}
            for c in FP_SWEEP:
                net = sum(f["scorers"][s]["cost_abort_matched"][c]["net_saved_min"] for f in r["rolling_origin"])
                n_builds = sum(f["n_test"] for f in r["rolling_origin"])
                abort[c] = {"net_saved_min": net, "net_per_build_min": net / n_builds,
                            "mitigation_pct": float(np.mean([f["scorers"][s]["cost_abort_matched"][c]["mitigation_pct"] for f in r["rolling_origin"]])),
                            "precision_at_operating_point": float(np.mean([f["scorers"][s]["cost_abort_matched"][c]["precision_at_operating_point"] for f in r["rolling_origin"]]))}
            be = [f["scorers"][s]["cost_abort_matched"]["2.0"]["break_even_fp_cost_min"] for f in r["rolling_origin"]]
            be = [b for b in be if b is not None]
            abort["break_even_fp_cost_min_median"] = float(np.median(be)) if be else None
            row["metrics"][s]["cost_abort"] = abort
            skip = {}
            for c in MISS_SWEEP:
                skip[c] = {
                    "net_saved_min": sum(f["scorers"][s]["cost_skip_matched"][c]["net_saved_min"] for f in r["rolling_origin"]),
                    "compute_saved_min": sum(f["scorers"][s]["cost_skip_matched"][c]["compute_saved_min"] for f in r["rolling_origin"]),
                    "failures_missed_pct": float(np.mean([f["scorers"][s]["cost_skip_matched"][c]["failures_missed_pct"] for f in r["rolling_origin"]])),
                    "skipped_pct": float(np.mean([f["scorers"][s]["cost_skip_matched"][c]["skipped_pct"] for f in r["rolling_origin"]])),
                }
            row["metrics"][s]["cost_skip"] = skip
        row["mean_wall_minutes"] = float(np.mean([f["context"]["mean_wall_minutes"] for f in r["rolling_origin"]]))
        row["failing_job_minutes"] = sum(f["context"]["failing_job_minutes"] for f in r["rolling_origin"])
        row["total_job_minutes"] = sum(f["context"]["total_job_minutes"] for f in r["rolling_origin"])
        # Ablation means over folds.
        ab = {}
        if r["rolling_origin"][0].get("ablation"):
            for subset in ABLATION_SUBSETS:
                if subset == "all":
                    ab[subset] = {s: row["metrics"][s]["auroc"] for s in LEARNED}
                    ab[subset + "_lift"] = {s: row["metrics"][s]["auprc_lift"] for s in LEARNED}
                    continue
                ab[subset] = {}
                ab[subset + "_lift"] = {}
                for s in r["rolling_origin"][0]["ablation"][subset]:
                    ab[subset][s] = float(np.mean([f["ablation"][subset][s]["auroc"] for f in r["rolling_origin"]]))
                    ab[subset + "_lift"][s] = float(np.mean([f["ablation"][subset][s]["auprc_lift"] for f in r["rolling_origin"]]))
        row["ablation"] = ab
        row["comparisons"] = r["comparisons"]
        row["block_length_sensitivity"] = r["block_length_sensitivity"]
        row["legacy_autoencoder"] = r["legacy_autoencoder"]
        row["ae_diagnostics"] = r["rolling_origin"][-1]["ae_diagnostics"]
        strict = r.get("label_sensitivity_strict") or {}
        if strict.get("folds"):
            row["strict"] = {"prevalence": strict["prevalence"], "builds_relabelled": strict["builds_relabelled"],
                             "auroc": {s: float(np.mean([f["scorers"][s]["auroc"] for f in strict["folds"]])) for s in SCORERS},
                             "auprc_lift": {s: float(np.mean([f["scorers"][s]["auprc_lift"] for f in strict["folds"]])) for s in SCORERS}}
        else:
            row["strict"] = {"prevalence": strict.get("prevalence"), "builds_relabelled": strict.get("builds_relabelled", 0), "auroc": {}, "auprc_lift": {}}
        rows.append(row)
    return rows


def pooled(rows, key="auroc"):
    out = {}
    for s in SCORERS:
        vals = [r["metrics"][s][key] for r in rows]
        ci = mean_ci(vals)
        out[s] = {**ci, "values": vals,
                  "projects_above_0_65": int(sum(1 for v in vals if v > 0.65)),
                  "projects_above_0_5": int(sum(1 for v in vals if v > 0.5)),
                  "sign_test_above_0_5": sign_test_above(vals, 0.5) if key == "auroc" else sign_test_above(vals, 1.0)}
    return out


def tally(rows, ref, metric):
    out = {}
    for s in SCORERS:
        if s == ref:
            continue
        wins = losses = ties = 0
        for r in rows:
            t = r["comparisons"][ref][metric][s]
            rej = t.get("holm", {}) and t["holm"]["reject_at_alpha"]
            if not rej:
                ties += 1
            elif t["observed_difference"] > 0:
                wins += 1
            else:
                losses += 1
        out[s] = {"wins": wins, "losses": losses, "ties": ties}
    return out


# ----------------------------------------------------------------------------
# LaTeX writers
# ----------------------------------------------------------------------------

def table(lines, path):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_corpus(rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2pt}",
         r"\caption{Project Corpus Summary}",
         r"\label{tab:corpus}",
         r"\begin{tabular}{p{1.55cm}rlrrrr}", r"\toprule",
         r"Project & Rank & Lang. & Builds & Rate & Strict & Overlap \\", r"\midrule"]
    for role in ["primary", "robustness"]:
        sub = [r for r in rows if r["role"] == role]
        if role == "robustness":
            L.append(r"\midrule")
        for r in sub:
            project = esc(r["project"]).replace("/", r"/\allowbreak ").replace(r"\_", r"\allowbreak\_")
            L.append(f"{project} & {r['rank']} & {esc(r['language'])} & {r['builds']:,} & {r['failure_rate_pct']:.1f}\\% & "
                     f"{r['strict_failure_rate_pct']:.1f}\\% & {r['concurrency']['prev_unfinished_pct']:.0f}\\% \\\\".replace(",", "{,}"))
    prim = [r for r in rows if r["role"] == "primary"]
    tb = sum(r["builds"] for r in prim)
    tf = sum(round(r["builds"] * r["failure_rate_pct"] / 100) for r in prim)
    L.append(r"\midrule")
    L.append((f"\\textbf{{Primary total}} & & & {tb:,} & {tf / tb * 100:.1f}\\% & & \\\\").replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)


def write_metric_table(rows, path, key, caption, label, nd=3):
    prim = [r for r in rows if r["role"] == "primary"]
    if key == "auroc":
        L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{p{1.35cm}rp{2.15cm}p{3.05cm}}", r"\toprule",
             r"Project & Prev. & Rules: PREV / LAST5 / HIST-LR & Detectors: RF / GB / AE / IF / PCA \\", r"\midrule"]
        for r in prim:
            best = max(r["metrics"][s][key] for s in SCORERS)
            def cell(s):
                v = r["metrics"][s][key]
                out = fmt(v, nd)
                return rf"\textbf{{{out}}}" if abs(v - best) < 1e-9 else out
            rules = " / ".join(cell(s) for s in ("PREV", "LAST5", "HIST-LR"))
            detectors = " / ".join(cell(s) for s in ("RF", "GB", "AE", "IF", "PCA"))
            project = esc(short_project(r["project"])).replace(r"\_", r"\allowbreak\_")
            L.append(f"{project} & {r['test_prevalence_mean']:.3f} & {rules} & {detectors} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        table(L, path)
        return
    L = [r"\begin{table*}[htbp]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{lr" + "r" * len(SCORERS) + "}", r"\toprule",
         r"Project & Prev. & " + " & ".join(SCORERS) + r" \\", r"\midrule"]
    for r in prim:
        cells = []
        best = max(r["metrics"][s][key] for s in SCORERS)
        for s in SCORERS:
            v = r["metrics"][s][key]
            cell = fmt(v, nd)
            if abs(v - best) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        L.append(f"{esc(short_project(r['project']))} & {r['test_prevalence_mean']:.3f} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    means = [fmt(float(np.mean([r["metrics"][s][key] for r in prim])), nd) for s in SCORERS]
    L.append(r"\textbf{Mean} & " + f"{np.mean([r['test_prevalence_mean'] for r in prim]):.3f} & " + " & ".join(means) + r" \\")
    meds = [fmt(float(np.median([r["metrics"][s][key] for r in prim])), nd) for s in SCORERS]
    L.append(r"\textbf{Median} & & " + " & ".join(meds) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    table(L, path)


def write_pooled(rows, path, pooled_auroc, pooled_lift):
    prim = [r for r in rows if r["role"] == "primary"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{1.5pt}",
         r"\caption{Pooled Performance Summary}",
         r"\label{tab:pooled}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Scorer & \shortstack{AUROC\\95\% CI} & Median & \shortstack{$>$0.5\\n / p} & \shortstack{$>$0.65\\n} & \shortstack{Lift\\95\% CI} & Norm. AP & Wilcox. \\", r"\midrule"]
    for s in SCORERS:
        a = pooled_auroc[s]
        l = pooled_lift[s]
        norm = float(np.mean([r["metrics"][s]["auprc_normalised"] for r in prim]))
        w = wilcoxon_paired([r["metrics"][s]["auroc"] for r in prim], [r["metrics"]["LAST5"]["auroc"] for r in prim])
        wtxt = "ref." if s == "LAST5" else format_p(w["p_value"])
        auroc_cell = rf"\shortstack{{{a['mean']:.3f}\\{{[{a['ci_low']:.3f}, {a['ci_high']:.3f}]}}}}"
        lift_cell = rf"\shortstack{{{l['mean']:.2f}\\{{[{l['ci_low']:.2f}, {l['ci_high']:.2f}]}}}}"
        L.append(f"{LABEL[s]} & {auroc_cell} & {a['median']:.3f} & "
                 f"{a['projects_above_0_5']}/10; {format_p(a['sign_test_above_0_5']['p_value'])} & {a['projects_above_0_65']}/10 & "
                 f"{lift_cell} & {norm:.3f} & {wtxt} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)


def write_tally(rows, path):
    prim = [r for r in rows if r["role"] == "primary"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2pt}",
         r"\caption{Paired Test Outcomes}",
         r"\label{tab:tally}",
         r"\begin{tabular}{p{1.35cm}cccc}", r"\toprule",
         r"Scorer & \shortstack{LAST5\\AUROC\\W/L/T; $\Delta$} & \shortstack{LAST5\\AUPRC\\W/L/T; $\Delta$} & \shortstack{RF\\AUROC\\W/L/T; $\Delta$} & \shortstack{RF\\AUPRC\\W/L/T; $\Delta$} \\", r"\midrule"]
    for s in SCORERS:
        cells = []
        for ref in ["LAST5", "RF"]:
            for metric in ["auroc", "auprc"]:
                if s == ref:
                    cells.append("ref.")
                    continue
                t = tally(prim, ref, metric)[s]
                d = float(np.mean([r["comparisons"][ref][metric][s]["observed_difference"] for r in prim]))
                cells.append(f"{t['wins']}/{t['losses']}/{t['ties']}; {fmt(d, 3, signed=True)}")
        L.append(f"{LABEL[s]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)


def write_drift(rows, path):
    prim = [r for r in rows if r["role"] == "primary"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Prevalence Across Folds}",
         r"\label{tab:drift}",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"Project & Train$_0$ & Train$_5$ & F0 & F1 & F2 & F3 & F4 & F5 & Range \\", r"\midrule"]
    for r in prim:
        fp = r["fold_prevalence"]
        L.append(f"{esc(short_project(r['project']))} & {r['fold_train_prevalence'][0]:.2f} & {r['fold_train_prevalence'][-1]:.2f} & "
                 + " & ".join(f"{v:.2f}" for v in fp) + f" & {max(fp) - min(fp):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)


def write_ablation(rows, path):
    prim = [r for r in rows if r["role"] == "primary" and r["ablation"]]
    L = [r"\begin{table*}[htbp]", r"\centering", r"\small",
         r"\caption{Feature Group Ablation}",
         r"\label{tab:ablation}",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Configuration & Features & PREV & LAST5 & HIST-LR & AE & IF & PCA & RF & GB \\".replace(r"& Features ", "& \\#Feat. "), r"\midrule"]
    counts = {"history_only": 5, "telemetry_only": 14, "churn_only": 19, "repository_only": 6, "no_repository": 38, "no_history": 39, "all": 44}
    for subset in ABLATION_SUBSETS:
        cells = []
        for s in SCORERS:
            vals = [r["ablation"][subset].get(s) for r in prim if subset in r["ablation"]]
            vals = [v for v in vals if v is not None]
            if subset == "all" and s in ("PREV", "LAST5", "HIST-LR"):
                vals = [r["metrics"][s]["auroc"] for r in prim]
            cells.append(fmt(float(np.mean(vals)), 3) if vals else "")
        L.append(f"{ABLATION_LABEL[subset]} & {counts[subset]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    # header fix: 10 columns
    L[5] = r"\begin{tabular}{lrrrrrrrrr}"
    table(L, path)


def write_operating(rows, path):
    prim = [r for r in rows if r["role"] == "primary"]
    L = [r"\begin{table*}[htbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Threshold Operating Points}",
         r"\label{tab:operating}",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{Native threshold} & \multicolumn{2}{c}{Matched budget (10\%)} & \multicolumn{3}{c}{Rolling threshold} \\",
         r"\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}",
         r"Scorer & Alarm rate & Precision & Recall & Degenerate & Precision & Recall & Alarm rate & Precision & Recall \\", r"\midrule"]
    for s in SCORERS:
        m = lambda k: float(np.mean([r["metrics"][s][k] for r in prim]))
        deg = int(sum(r["metrics"][s]["native_degenerate_folds"] for r in prim))
        roll = ""
        if "rolling_alarm_rate" in prim[0]["metrics"][s]:
            roll = f"{m('rolling_alarm_rate'):.3f} & {m('rolling_precision'):.3f} & {m('rolling_recall'):.3f}"
        else:
            roll = " & & "
        L.append(f"{LABEL[s]} & {m('native_alarm_rate'):.3f} & {m('native_precision'):.3f} & {m('native_recall'):.3f} & {deg} & "
                 f"{m('matched_precision'):.3f} & {m('matched_recall'):.3f} & {roll} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    table(L, path)


def write_cost_sweep(rows, path):
    prim = [r for r in rows if r["role"] == "primary"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{1.5pt}",
         r"\caption{Withhold Cost Sensitivity}",
         r"\label{tab:cost_sweep}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Scorer & 2 min & 5 min & 15 min & 30 min & 1 build & \shortstack{Positive\\at 2 min} & \shortstack{Positive\\at 1 build} \\", r"\midrule"]
    for s in SCORERS:
        cells = []
        for c in FP_SWEEP:
            vals = [r["metrics"][s]["cost_abort"][c]["net_per_build_min"] for r in prim]
            cells.append(fmt(float(np.median(vals)), 2))
        pos2 = sum(1 for r in prim if r["metrics"][s]["cost_abort"]["2.0"]["net_saved_min"] > 0)
        posb = sum(1 for r in prim if r["metrics"][s]["cost_abort"]["one_build"]["net_saved_min"] > 0)
        L.append(f"{LABEL[s]} & " + " & ".join(cells) + f" & {pos2}/10 & {posb}/10 \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)


def write_cost_per_project(rows, path, scorers=("LAST5", "RF", "AE")):
    prim = [r for r in rows if r["role"] == "primary"]

    def money(v):
        v = 0.0 if abs(v) < 0.5 else v
        return f"{v:,.0f}".replace(",", "{,}")

    L = [r"\begin{table*}[htbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{2.4pt}",
         r"\caption{Project Withhold Outcomes}",
         r"\label{tab:cost_project}",
         r"\begin{tabular}{lrr" + "rrr" * len(scorers) + "}", r"\toprule",
         r"Project & Build & Prec. & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{s}}}" for s in scorers) + r" \\",
         " ".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(scorers))),
         r" & (min) & LAST5 & " + " & ".join("Net 2 min & Net 1 build & Break even" for _ in scorers) + r" \\", r"\midrule"]
    for r in prim:
        cells = []
        for s in scorers:
            ca = r["metrics"][s]["cost_abort"]
            be = ca["break_even_fp_cost_min_median"]
            if be is None:
                be_txt = "n/a"
            elif be > 9999:
                be_txt = r"$>$9{,}999"
            else:
                be_txt = money(be)
            cells += [money(ca["2.0"]["net_saved_min"]), money(ca["one_build"]["net_saved_min"]), be_txt]
        prec = r["metrics"]["LAST5"]["cost_abort"]["2.0"]["precision_at_operating_point"]
        L.append(f"{esc(short_project(r['project']))} & {r['mean_wall_minutes']:.0f} & {prec:.2f} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    table(L, path)


def write_skip(rows, path):
    skip_path = ROOT / "results" / "skip_scenario.json"
    if not skip_path.exists():
        return
    payload = json.loads(skip_path.read_text(encoding="utf-8"))
    budget = payload["skip_budget"]
    primary = {r["project"] for r in rows if r["role"] == "primary"}
    per_scorer = {s: {"missed": [], "saved_pct": [], **{c: [] for c in MISS_SWEEP}} for s in SCORERS}
    for proj in payload["projects"]:
        if proj["project"] not in primary:
            continue
        total = sum(f["total_job_minutes"] for f in proj["folds"])
        for s in SCORERS:
            missed = float(np.mean([f["scorers"][s]["30.0"]["failures_missed_pct"] for f in proj["folds"]]))
            saved = sum(f["scorers"][s]["30.0"]["compute_saved_min"] for f in proj["folds"])
            per_scorer[s]["missed"].append(missed)
            per_scorer[s]["saved_pct"].append(saved / total * 100 if total else 0.0)
            for c in MISS_SWEEP:
                net = sum(f["scorers"][s][c]["net_saved_min"] for f in proj["folds"])
                n_builds = sum(f["n_test"] for f in proj["folds"])
                per_scorer[s][c].append(net / n_builds)
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2pt}",
         r"\caption{Build Skip Outcomes}",
         r"\label{tab:skip}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Scorer & Missed & Saved & Net: 30m & Net: 120m & Net: 1 build \\", r"\midrule"]
    for s in SCORERS:
        d = per_scorer[s]
        L.append(f"{LABEL[s]} & {np.mean(d['missed']):.1f}\\% & {np.mean(d['saved_pct']):.1f}\\% & "
                 + " & ".join(fmt(float(np.median(d[c])), 2) for c in MISS_SWEEP) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)
    return {s: {"missed_pct": float(np.mean(v["missed"])), "saved_pct": float(np.mean(v["saved_pct"])),
                **{c: float(np.median(v[c])) for c in MISS_SWEEP}} for s, v in per_scorer.items()}


def write_robustness(rows, path):
    prim = [r for r in rows if r["role"] == "primary"]
    allr = rows
    rob = [r for r in rows if r["role"] == "robustness"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{1.5pt}",
         r"\caption{Robustness Checks}",
         r"\label{tab:robustness}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Scorer & Primary 10 & Omitted 4 & All 14 & Single cut & Strict label \\",
         r"\multicolumn{6}{l}{\scriptsize Cells report AUROC / AUPRC lift.} \\",
         r"\midrule"]
    for s in SCORERS:
        def m(rs, k):
            return float(np.mean([r["metrics"][s][k] for r in rs]))
        strict_a = [r["strict"]["auroc"].get(s) for r in prim if r["strict"]["auroc"]]
        strict_l = [r["strict"]["auprc_lift"].get(s) for r in prim if r["strict"]["auprc_lift"]]
        # projects with no relabelled builds keep their lenient numbers
        for r in prim:
            if not r["strict"]["auroc"]:
                strict_a.append(r["metrics"][s]["auroc"])
                strict_l.append(r["metrics"][s]["auprc_lift"])
        L.append(f"{LABEL[s]} & {m(prim, 'auroc'):.3f}/{m(prim, 'auprc_lift'):.2f} & {m(rob, 'auroc'):.3f}/{m(rob, 'auprc_lift'):.2f} & "
                 f"{m(allr, 'auroc'):.3f}/{m(allr, 'auprc_lift'):.2f} & {m(prim, 'single_cut_auroc'):.3f}/{m(prim, 'single_cut_lift'):.2f} & "
                 f"{np.mean(strict_a):.3f}/{np.mean(strict_l):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)


def write_block(rows, path):
    prim = [r for r in rows if r["role"] == "primary"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Bootstrap Block Sensitivity}",
         r"\label{tab:block}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Comparison & $L = 1$ & $L = 5$ & $L = 10$ & $L = 20$ & $L = 40$ \\", r"\midrule"]
    for key, lab in [("RF_vs_LAST5", "RF vs LAST5"), ("GB_vs_LAST5", "GB vs LAST5"), ("AE_vs_RF", "AE vs RF")]:
        cells = []
        for L_ in ["1", "5", "10", "20", "40"]:
            widths = [r["block_length_sensitivity"][key][L_]["ci_high"] - r["block_length_sensitivity"][key][L_]["ci_low"] for r in prim
                      if r["block_length_sensitivity"][key][L_]["ci_high"] is not None]
            sig = sum(1 for r in prim if (r["block_length_sensitivity"][key][L_]["p_value"] or 1) < 0.05)
            cells.append(f"{np.mean(widths):.3f} ({sig})")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table(L, path)


def write_ae_repair(rows, path):
    prim = [r for r in rows if r["role"] == "primary"]
    L = [r"\begin{table*}[htbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{2.4pt}",
         r"\caption{Autoencoder Weighting Repair}",
         r"\label{tab:ae_repair}",
         r"\begin{tabular}{lrlrlrrr}", r"\toprule",
         r"Project & Zero var. & Legacy top feature & Share & Repaired top feature & Share & AUROC legacy & AUROC repaired \\", r"\midrule"]
    for r in prim:
        lg = r["legacy_autoencoder"]
        rp = r["ae_diagnostics"]
        rep_auroc = r["metrics"]["AE"]["fold_auroc"][-1]
        L.append(f"{esc(short_project(r['project']))} & {lg['zero_variance_features']} & {esc(lg['top_feature'])} & {lg['top_feature_share_pct']:.0f}\\% & "
                 f"{esc(rp['top_feature'])} & {rp['top_feature_share_pct']:.0f}\\% & {lg['auroc']:.3f} & {rep_auroc:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    table(L, path)


def write_summary_md(rows, pooled_auroc, pooled_lift, path):
    prim = [r for r in rows if r["role"] == "primary"]
    lines = ["# Cross project summary (rolling origin, ten primary projects)", "",
             "| Scorer | Mean AUROC | 95% CI | Median | Projects > 0.65 | Mean lift | Norm. AUPRC |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in SCORERS:
        a, l = pooled_auroc[s], pooled_lift[s]
        norm = float(np.mean([r["metrics"][s]["auprc_normalised"] for r in prim]))
        lines.append(f"| {LABEL[s]} | {a['mean']:.3f} | [{a['ci_low']:.3f}, {a['ci_high']:.3f}] | {a['median']:.3f} | {a['projects_above_0_65']}/10 | {l['mean']:.2f} | {norm:.3f} |")
    lines += ["", "## Per project mean AUROC", "", "| Project | Prev. | " + " | ".join(SCORERS) + " |", "| :--- | ---: | " + " | ".join("---:" for _ in SCORERS) + " |"]
    for r in rows:
        lines.append(f"| {r['project']} ({r['role']}) | {r['test_prevalence_mean']:.3f} | " + " | ".join(f"{r['metrics'][s]['auroc']:.3f}" for s in SCORERS) + " |")
    lines += ["", "## Holm tallies (wins/losses/ties) vs LAST5 and vs RF, AUROC then AUPRC", ""]
    for ref in ["LAST5", "RF"]:
        for metric in ["auroc", "auprc"]:
            t = tally(prim, ref, metric)
            lines.append(f"- vs {ref}, {metric}: " + ", ".join(f"{s} {v['wins']}/{v['losses']}/{v['ties']}" for s, v in t.items()))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results_dir = ROOT / "results"
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit("No result files. Run scripts/run_experiment.py first.")
    rows = per_project_table(runs)
    prim = [r for r in rows if r["role"] == "primary"]
    pooled_auroc = pooled(prim, "auroc")
    pooled_lift = pooled(prim, "auprc_lift")
    summary = {
        "n_projects": len(rows), "n_primary": len(prim),
        "per_project": rows,
        "pooled_primary": {"auroc": pooled_auroc, "auprc_lift": pooled_lift,
                           "auprc_normalised": pooled(prim, "auprc_normalised")},
        "pooled_all14": {"auroc": pooled(rows, "auroc"), "auprc_lift": pooled(rows, "auprc_lift")},
        "tallies": {ref: {metric: tally(prim, ref, metric) for metric in ["auroc", "auprc"]} for ref in ["LAST5", "RF"]},
    }
    (results_dir / "cross_project_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else float(o)), encoding="utf-8")

    write_corpus(rows, docs / "corpus_table.tex")
    write_metric_table(rows, docs / "auroc_table.tex", "auroc",
                       "Project AUROC Results", "tab:auroc")
    write_metric_table(rows, docs / "lift_table.tex", "auprc_lift",
                       "Project AUPRC Lift", "tab:lift", nd=2)
    write_pooled(rows, docs / "pooled_table.tex", pooled_auroc, pooled_lift)
    write_tally(rows, docs / "tally_table.tex")
    write_drift(rows, docs / "drift_table.tex")
    write_ablation(rows, docs / "ablation_table.tex")
    write_operating(rows, docs / "operating_table.tex")
    write_cost_sweep(rows, docs / "cost_sweep_table.tex")
    write_cost_per_project(rows, docs / "cost_project_table.tex")
    summary["skip_scenario"] = write_skip(rows, docs / "skip_table.tex")
    (results_dir / "cross_project_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else float(o)), encoding="utf-8")
    write_robustness(rows, docs / "robustness_table.tex")
    write_block(rows, docs / "block_table.tex")
    write_ae_repair(rows, docs / "ae_repair_table.tex")
    write_summary_md(rows, pooled_auroc, pooled_lift, docs / "summary.md")
    print(f"Aggregated {len(rows)} projects ({len(prim)} primary). Tables written to {docs}.")


if __name__ == "__main__":
    main()
