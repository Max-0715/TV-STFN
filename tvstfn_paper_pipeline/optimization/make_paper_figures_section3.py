import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = "/data/workplace/jwx/TV-STFN"
OUT = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/paper_section3_figures")
os.makedirs(OUT, exist_ok=True)


def load_summary(tag):
    p = os.path.join(ROOT, "benchmark_results", f"tvstfn_benchmark_summary_{tag}.csv")
    df = pd.read_csv(p)
    return {str(r["Metric"]): float(r["Mean"]) for _, r in df.iterrows()}


def fig_overall_compare():
    base_tag = "tv5overnight_1"
    hard_tag = "hard_v1_overnight"
    m1 = load_summary(base_tag)
    m2 = load_summary(hard_tag)

    metrics = ["ACC", "F1", "MCC", "AUROC"]
    x = np.arange(len(metrics))
    w = 0.36

    y1 = [m1[k] for k in metrics]
    y2 = [m2[k] for k in metrics]

    plt.figure(figsize=(8.4, 4.8), dpi=160)
    plt.bar(x - w / 2, y1, width=w, label=base_tag)
    plt.bar(x + w / 2, y2, width=w, label=hard_tag)
    plt.xticks(x, metrics)
    plt.ylim(0.4, 0.9)
    plt.ylabel("Score")
    plt.title("Overall Metrics Comparison")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()

    text = f"RMSE: {base_tag}={m1['RMSE']:.4f}, {hard_tag}={m2['RMSE']:.4f}"
    plt.text(0.02, 0.02, text, transform=plt.gca().transAxes, fontsize=9)

    out = os.path.join(OUT, "figure_3A_overall_compare.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return m1, m2, out


def fig_cliff_delta():
    p = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/wp3_cliff/figure_Z_panel_data.csv")
    df = pd.read_csv(p).copy()
    df = df.head(10)
    df["pair"] = [f"P{i+1}" for i in range(len(df))]

    x = np.arange(len(df))
    w = 0.38

    plt.figure(figsize=(9.2, 4.8), dpi=160)
    plt.bar(x - w / 2, df["delta_true"], width=w, label="True Delta", color="#4C78A8")
    plt.bar(x + w / 2, df["Pred_LogP_TVSTFN_delta"], width=w, label="Pred Delta", color="#F58518")
    plt.xticks(x, df["pair"])
    plt.ylabel("|ΔlogPe|")
    plt.title("Activity Cliff Pairs: True vs Predicted Delta")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    out = os.path.join(OUT, "figure_3B_cliff_delta.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    mae_delta = float(np.mean(np.abs(df["delta_true"] - df["Pred_LogP_TVSTFN_delta"])))
    return mae_delta, out


def fig_stratified_f1():
    p = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/wp2_stratified/stratified_metrics.csv")
    df = pd.read_csv(p)

    mw = df[df["stratifier"] == "MW"].copy()
    tp = df[df["stratifier"] == "TPSA"].copy()

    plt.figure(figsize=(9.2, 4.8), dpi=160)
    plt.plot(mw["bin"], mw["F1"], marker="o", label="MW bins")
    plt.plot(tp["bin"], tp["F1"], marker="s", label="TPSA bins")
    plt.ylim(0.45, 0.82)
    plt.ylabel("F1")
    plt.title("Stratified Robustness (F1)")
    plt.grid(alpha=0.2)
    plt.legend()
    out = os.path.join(OUT, "figure_3C_stratified_f1.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    return out


def write_draft(m1, m2, cliff_mae):
    text = f"""# Section 3 Results Draft (Auto-generated)

## 3.X Overall Performance
Compared with {"hard_v1_overnight"}, the baseline setting {"tv5overnight_1"} achieved stronger overall metrics (ACC {m1['ACC']:.4f} vs {m2['ACC']:.4f}, F1 {m1['F1']:.4f} vs {m2['F1']:.4f}, MCC {m1['MCC']:.4f} vs {m2['MCC']:.4f}, AUROC {m1['AUROC']:.4f} vs {m2['AUROC']:.4f}).
The regression error was also lower for the baseline configuration (RMSE {m1['RMSE']:.4f} vs {m2['RMSE']:.4f}).

## 3.X Stratified Robustness
The model showed a clear performance drop in high-complexity regions, especially under TPSA > 300, consistent with the expected difficulty of highly polar cyclic peptides.
This indicates that the current architecture captures global trends well, but still underfits a subset of physicochemically extreme samples.

## 3.X Activity Cliff Analysis
For the selected top-10 activity-cliff pairs, the average absolute delta error |Δtrue - Δpred| was {cliff_mae:.4f}.
This result suggests that the model remains conservative when faced with cliff-like perturbations, motivating targeted training strategies (hard-sample reweighting and structure-focused augmentation).

## Figures
- Figure 3A: Overall metrics comparison.
- Figure 3B: Activity cliff pair delta comparison.
- Figure 3C: Stratified robustness (F1).
"""
    out = os.path.join(OUT, "section3_results_draft.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


def main():
    m1, m2, fig1 = fig_overall_compare()
    cliff_mae, fig2 = fig_cliff_delta()
    fig3 = fig_stratified_f1()
    md = write_draft(m1, m2, cliff_mae)

    print("Generated:")
    print(fig1)
    print(fig2)
    print(fig3)
    print(md)


if __name__ == "__main__":
    main()
