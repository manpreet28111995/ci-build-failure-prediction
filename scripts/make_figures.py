#!/usr/bin/env python3
"""Regenerate every figure in the paper from results/cross_project_summary.json
and the per project score archives results/<project>_scores.npz.

Figures (all written to figures/ as PNG at 300 dpi):
  fig_prevalence_drift.png    test window prevalence per project and fold
  fig_concurrency.png         share of builds whose predecessor was still running
  fig_auroc_heatmap.png       mean AUROC per project and scorer
  fig_lift_dots.png           AUPRC lift per project: baselines against learned detectors
  fig_fold_trajectories.png   AUROC per fold for three scorers, one panel per project
  fig_ablation.png            feature group ablation for RF, GB and AE
  fig_ae_repair.png           legacy against repaired autoencoder score distributions
  fig_cost_sweep.png          net saving per build against false alarm cost
  fig_block_sensitivity.png   interval width against block length
  fig_roc_pr.png              ROC and precision recall curves for one project
  fig_robustness.png          pooled AUROC for the 10, the 4 and the 14 projects
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_pdm")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCORERS = ["PREV", "LAST5", "HIST-LR", "AE", "IF", "PCA", "RF", "GB"]
LABEL = {"PREV": "Previous outcome", "LAST5": "Last five rate", "HIST-LR": "History LR",
         "AE": "Autoencoder", "IF": "Isolation forest", "PCA": "PCA reconstruction",
         "RF": "Random forest", "GB": "Gradient boosting"}
# Fixed categorical assignment (validated palette, slots 1 to 8). Colour follows the scorer.
COLOR = {"PREV": "#2a78d6", "LAST5": "#eb6834", "HIST-LR": "#1baf7a", "AE": "#eda100",
         "IF": "#e87ba4", "PCA": "#008300", "RF": "#4a3aa7", "GB": "#e34948"}
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "normal", "axes.titlesize": 9.5,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def short(name: str) -> str:
    return name.split("/")[-1]


def load():
    summary = json.loads((ROOT / "results" / "cross_project_summary.json").read_text(encoding="utf-8"))
    return summary


def save(fig, name):
    out = ROOT / "figures" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_prevalence_drift(rows):
    prim = [r for r in rows if r["role"] == "primary"]
    mat = np.array([r["fold_prevalence"] for r in prim])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.8, aspect="auto")
    ax.set_yticks(range(len(prim)))
    ax.set_yticklabels([short(r["project"]) for r in prim])
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"fold {k}" for k in range(mat.shape[1])])
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.42 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Failure prevalence of the test window")
    cb.outline.set_visible(False)
    ax.set_title("Test window prevalence moves by up to 0.5 within a project")
    save(fig, "fig_prevalence_drift.png")


def fig_concurrency(rows):
    order = sorted(rows, key=lambda r: r["concurrency"]["prev_unfinished_pct"])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    y = np.arange(len(order))
    vals = [r["concurrency"]["prev_unfinished_pct"] for r in order]
    vals10 = [r["concurrency"]["any_of_last10_unfinished_pct"] for r in order]
    ax.barh(y + 0.2, vals10, height=0.36, color=SEQ[1], label="Any of the previous ten builds still running")
    ax.barh(y - 0.2, vals, height=0.36, color=SEQ[4], label="Previous build still running")
    ax.set_yticks(y)
    ax.set_yticklabels([short(r["project"]) + (" *" if r["role"] == "robustness" else "") for r in order])
    ax.set_xlabel("Share of builds (percent)")
    ax.set_xlim(0, 100)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi - 0.2, f"{v:.0f}", va="center", fontsize=7.5, color=TEXT2)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("How often the previous outcome is not yet known at build start")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_concurrency.png")


def fig_auroc_heatmap(rows):
    prim = [r for r in rows if r["role"] == "primary"]
    mat = np.array([[r["metrics"][s]["auroc"] for s in SCORERS] for r in prim])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=0.85)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(range(len(prim)))
    ax.set_yticklabels([short(r["project"]) for r in prim])
    ax.set_xticks(range(len(SCORERS)))
    ax.set_xticklabels(SCORERS)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(mat[i, j] - 0.5) > 0.2 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Mean AUROC over six rolling windows")
    cb.outline.set_visible(False)
    ax.set_title("Mean AUROC per project: grey is chance, blue above, red below")
    save(fig, "fig_auroc_heatmap.png")


def fig_lift_dots(rows):
    prim = sorted([r for r in rows if r["role"] == "primary"], key=lambda r: r["test_prevalence_mean"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(prim))
    show = ["LAST5", "HIST-LR", "RF", "GB", "AE"]
    for i, s in enumerate(show):
        vals = [r["metrics"][s]["auprc_lift"] for r in prim]
        ax.plot(x, vals, marker="o", markersize=5, linewidth=1.4, color=COLOR[s], label=LABEL[s],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(1.0, color=TEXT2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(prim) - 0.6, 1.02, "random ranker", fontsize=7.5, color=TEXT2, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short(r['project'])} ({r['test_prevalence_mean']:.2f})" for r in prim], fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("AUPRC divided by test prevalence")
    ax.set_xlabel("Project (mean test window prevalence), ordered by prevalence")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.set_title("Lift over prevalence, projects ordered by prevalence")
    save(fig, "fig_lift_dots.png")


def fig_fold_trajectories(rows):
    prim = [r for r in rows if r["role"] == "primary"]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2), sharey=True)
    show = ["LAST5", "RF", "AE"]
    for ax, r in zip(axes.ravel(), prim):
        k = np.arange(r["n_folds"])
        for s in show:
            ax.plot(k, r["metrics"][s]["fold_auroc"], marker="o", markersize=4, linewidth=1.4, color=COLOR[s], label=LABEL[s])
        ax.axhline(0.5, color=TEXT2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax2 = ax.twinx() if False else None
        prev = r["fold_prevalence"]
        ax.bar(k, prev, width=0.6, color=GRID, zorder=0)
        ax.set_title(short(r["project"]), fontsize=8.5)
        ax.set_xticks(k)
        ax.set_xticklabels([str(i) for i in k], fontsize=7)
        ax.set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("AUROC / prevalence")
    axes[1, 0].set_ylabel("AUROC / prevalence")
    for ax in axes[1]:
        ax.set_xlabel("rolling origin fold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
    fig.suptitle("AUROC per rolling origin fold; the grey bars give the window's failure prevalence", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig_fold_trajectories.png")


def fig_ablation(rows):
    prim = [r for r in rows if r["role"] == "primary" and r["ablation"]]
    subsets = ["history_only", "telemetry_only", "churn_only", "repository_only", "no_repository", "no_history", "all"]
    labels = ["History\nonly", "Telemetry\nonly", "Churn\nonly", "Repository\nonly", "All minus\nrepository", "All minus\nhistory", "All four\ngroups"]
    show = ["RF", "GB", "AE"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(subsets))
    w = 0.26
    for i, s in enumerate(show):
        means = [np.mean([r["ablation"][sub][s] for r in prim if sub in r["ablation"] and s in r["ablation"][sub]]) for sub in subsets]
        ax.bar(x + (i - 1) * w, means, width=w - 0.03, color=COLOR[s], label=LABEL[s])
        for xi, m in zip(x + (i - 1) * w, means):
            ax.text(xi, m + 0.004, f"{m:.2f}", ha="center", fontsize=6.5, color=TEXT2)
    ax.axhline(0.5, color=TEXT2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.45, 0.72)
    ax.set_ylabel("Mean AUROC over ten projects")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.set_title("Feature group ablation: history carries the signal, repository attributes remove it")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_ablation.png")


def fig_ae_repair(rows, project_stem="opf_openproject"):
    prim = [r for r in rows if r["role"] == "primary"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    y = np.arange(len(prim))
    leg = [r["legacy_autoencoder"]["top_feature_share_pct"] for r in prim]
    rep = [r["ae_diagnostics"]["top_feature_share_pct"] for r in prim]
    ax.barh(y + 0.2, leg, height=0.36, color="#e34948", label="Legacy inverse-variance weighting")
    ax.barh(y - 0.2, rep, height=0.36, color="#2a78d6", label="Repaired weighting")
    ax.set_yticks(y)
    ax.set_yticklabels([short(r["project"]) for r in prim])
    ax.set_xlabel("Share of test reconstruction error carried by one feature (percent)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("One feature dominated the legacy score", fontsize=9)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    npz = np.load(ROOT / "results" / f"{project_stem}_scores.npz", allow_pickle=True)
    y_te = npz["y_test"]
    legacy = npz["legacy_ae_scores"]
    repaired = npz["score_AE"]
    thr = float(npz["ae_native_threshold"][0])
    for scores, color, name in [(legacy, "#e34948", "legacy"), (repaired, "#2a78d6", "repaired")]:
        s = np.log10(np.maximum(scores, 1e-6))
        bins = np.linspace(s.min(), s.max(), 40)
        ax.hist(s[y_te == 0], bins=bins, density=True, histtype="step", linewidth=1.6, color=color, label=f"{name}: passing builds")
        ax.hist(s[y_te == 1], bins=bins, density=True, histtype="step", linewidth=1.6, linestyle=(0, (3, 2)), color=color, label=f"{name}: failing builds")
    ax.axvline(np.log10(thr), color=TEXT2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(np.log10(thr), ax.get_ylim()[1] * 0.55, " repaired native threshold", fontsize=7.5, color=TEXT2, va="top")
    ax.set_xlabel("log10 reconstruction error, final fold")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{project_stem.replace('_', '/')}: score distributions", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_ae_repair.png")


def fig_cost_sweep(rows):
    prim = [r for r in rows if r["role"] == "primary"]
    sweep = ["2.0", "5.0", "15.0", "30.0", "one_build"]
    xlab = ["2", "5", "15", "30", "one build"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    for s in ["LAST5", "HIST-LR", "RF", "GB", "AE"]:
        med = [np.median([r["metrics"][s]["cost_abort"][c]["net_per_build_min"] for r in prim]) for c in sweep]
        ax.plot(range(len(sweep)), med, marker="o", markersize=5, linewidth=1.5, color=COLOR[s], label=LABEL[s],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(0, color=TEXT2, linewidth=0.9)
    ax.set_xticks(range(len(sweep)))
    ax.set_xticklabels(xlab)
    ax.set_xlabel("Cost of a false alarm (minutes)")
    ax.set_ylabel("Median net saving per build (compute minutes)")
    ax.set_title("Withhold scenario, ten percent alert budget", fontsize=9)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for s in ["LAST5", "HIST-LR", "RF", "GB", "AE"]:
        vals = [r["metrics"][s]["cost_abort"]["one_build"]["net_per_build_min"] for r in prim]
        xs = np.arange(len(prim))
        ax.scatter(xs, vals, color=COLOR[s], s=26, label=LABEL[s], edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.axhline(0, color=TEXT2, linewidth=0.9)
    ax.set_xticks(np.arange(len(prim)))
    ax.set_xticklabels([short(r["project"]) for r in prim], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Net saving per build, false alarm = one build")
    ax.set_title("Per project outcome when a false alarm costs a full build", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_cost_sweep.png")


def fig_block_sensitivity(rows):
    prim = [r for r in rows if r["role"] == "primary"]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    Ls = ["1", "5", "10", "20", "40"]
    for key, lab, color in [("RF_vs_LAST5", "RF vs LAST5", COLOR["RF"]), ("GB_vs_LAST5", "GB vs LAST5", COLOR["GB"]), ("AE_vs_RF", "AE vs RF", COLOR["AE"])]:
        widths = [np.mean([r["block_length_sensitivity"][key][L]["ci_high"] - r["block_length_sensitivity"][key][L]["ci_low"] for r in prim]) for L in Ls]
        ax.plot(range(len(Ls)), widths, marker="o", markersize=5, linewidth=1.5, color=color, label=lab, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels(Ls)
    ax.set_xlabel("Block length (builds); 1 is independent resampling")
    ax.set_ylabel("Mean 95 percent interval width (AUROC)")
    ax.set_title("Interval width against bootstrap block length")
    ax.legend(fontsize=8)
    save(fig, "fig_block_sensitivity.png")


def fig_roc_pr(rows, project_stem="jruby_jruby"):
    npz = np.load(ROOT / "results" / f"{project_stem}_scores.npz", allow_pickle=True)
    y = npz["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    show = ["LAST5", "HIST-LR", "RF", "AE"]
    for s in show:
        sc = npz[f"score_{s}"]
        fpr, tpr, _ = roc_curve(y, sc)
        prec, rec, _ = precision_recall_curve(y, sc)
        from sklearn.metrics import average_precision_score, roc_auc_score
        axes[0].plot(fpr, tpr, color=COLOR[s], linewidth=1.6, label=f"{LABEL[s]} ({roc_auc_score(y, sc):.2f})")
        axes[1].plot(rec, prec, color=COLOR[s], linewidth=1.6, label=f"{LABEL[s]} ({average_precision_score(y, sc):.2f})")
    axes[0].plot([0, 1], [0, 1], color=TEXT2, linewidth=0.9, linestyle=(0, (4, 3)))
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC, final fold (AUROC in legend)", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[1].axhline(y.mean(), color=TEXT2, linewidth=0.9, linestyle=(0, (4, 3)))
    axes[1].text(0.02, y.mean() + 0.02, f"prevalence {y.mean():.2f}", fontsize=7.5, color=TEXT2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Precision recall, final fold (AUPRC in legend)", fontsize=9)
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(project_stem.replace("_", "/"), fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig_roc_pr.png")


def fig_robustness(summary):
    rows = summary["per_project"]
    prim = [r for r in rows if r["role"] == "primary"]
    rob = [r for r in rows if r["role"] == "robustness"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(SCORERS))
    w = 0.26
    for i, (name, rs, color) in enumerate([("Primary ten", prim, SEQ[5]), ("Omitted four", rob, SEQ[2]), ("All fourteen", rows, SEQ[3])]):
        means = [np.mean([r["metrics"][s]["auroc"] for r in rs]) for s in SCORERS]
        ax.bar(x + (i - 1) * w, means, width=w - 0.03, color=color, label=name)
    ax.axhline(0.5, color=TEXT2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(SCORERS)
    ax.set_ylim(0.45, 0.7)
    ax.set_ylabel("Mean AUROC")
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Pooled AUROC is stable across the three samples")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_robustness.png")


def main():
    summary = load()
    rows = summary["per_project"]
    fig_prevalence_drift(rows)
    fig_concurrency(rows)
    fig_auroc_heatmap(rows)
    fig_lift_dots(rows)
    fig_fold_trajectories(rows)
    fig_ablation(rows)
    fig_ae_repair(rows)
    fig_cost_sweep(rows)
    fig_block_sensitivity(rows)
    fig_roc_pr(rows)
    fig_robustness(summary)
    print("Figures complete.")


if __name__ == "__main__":
    main()
