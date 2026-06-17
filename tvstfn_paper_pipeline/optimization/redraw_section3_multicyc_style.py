import os
import re
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


ROOT = "/data/workplace/jwx/TV-STFN"
OUT = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/paper_section3_figures")
os.makedirs(OUT, exist_ok=True)


STYLE = {
    "face": "#ffffff",
    "grid": "#d9d9d9",
    "text": "#1f1f1f",
    "baseline": "#3b6fb6",
    "hard": "#f2a65a",
    "true": "#3b6fb6",
    "pred": "#d96c75",
    "mw": "#4f7a69",
    "tpsa": "#8d6cab",
}


def set_global_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": STYLE["face"],
            "axes.facecolor": STYLE["face"],
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.8,
            "axes.labelcolor": STYLE["text"],
            "xtick.color": STYLE["text"],
            "ytick.color": STYLE["text"],
            "grid.color": STYLE["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def find_summary(tag: str) -> str:
    candidates = [
        os.path.join(ROOT, "benchmark_results", f"tvstfn_benchmark_summary_{tag}.csv"),
        os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/paper_v1", f"tvstfn_benchmark_summary_{tag}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def parse_metrics_from_draft() -> Tuple[Dict[str, float], Dict[str, float]]:
    draft = os.path.join(OUT, "section3_results_draft.md")
    if not os.path.exists(draft):
        raise FileNotFoundError("section3_results_draft.md not found and summary csv missing")

    text = open(draft, "r", encoding="utf-8").read()
    # Extract numbers from sentence with fixed order.
    m = re.search(
        r"ACC\s+([0-9.]+)\s+vs\s+([0-9.]+),\s+F1\s+([0-9.]+)\s+vs\s+([0-9.]+),\s+MCC\s+([0-9.]+)\s+vs\s+([0-9.]+),\s+AUROC\s+([0-9.]+)\s+vs\s+([0-9.]+)",
        text,
    )
    rm = re.search(r"RMSE\s+([0-9.]+)\s+vs\s+([0-9.]+)", text)
    if not m or not rm:
        raise ValueError("Failed to parse metrics from section3_results_draft.md")

    baseline = {
        "ACC": float(m.group(1)),
        "F1": float(m.group(3)),
        "MCC": float(m.group(5)),
        "AUROC": float(m.group(7)),
        "RMSE": float(rm.group(1)),
    }
    hard = {
        "ACC": float(m.group(2)),
        "F1": float(m.group(4)),
        "MCC": float(m.group(6)),
        "AUROC": float(m.group(8)),
        "RMSE": float(rm.group(2)),
    }
    return baseline, hard


def load_compare_metrics() -> Tuple[Dict[str, float], Dict[str, float]]:
    p1 = find_summary("tv5overnight_1")
    p2 = find_summary("hard_v1_overnight")
    if p1 and p2:
        d1 = pd.read_csv(p1)
        d2 = pd.read_csv(p2)
        m1 = {str(r["Metric"]): float(r["Mean"]) for _, r in d1.iterrows()}
        m2 = {str(r["Metric"]): float(r["Mean"]) for _, r in d2.iterrows()}
        return m1, m2
    return parse_metrics_from_draft()


def panel_a(ax):
    m1, m2 = load_compare_metrics()
    metrics = ["ACC", "F1", "MCC", "AUROC"]
    x = np.arange(len(metrics))
    w = 0.36

    y1 = [m1[k] for k in metrics]
    y2 = [m2[k] for k in metrics]

    ax.bar(x - w / 2, y1, width=w, color=STYLE["baseline"], alpha=0.9, label="tv5overnight_1")
    ax.bar(x + w / 2, y2, width=w, color=STYLE["hard"], alpha=0.9, label="hard_v1_overnight")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.02,
        0.04,
        f"RMSE: tv5={m1['RMSE']:.4f}, hard={m2['RMSE']:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        color="#404040",
    )


def panel_b(ax):
    p = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/wp3_cliff/figure_Z_panel_data.csv")
    df = pd.read_csv(p).head(10).copy()
    df["pair"] = [f"P{i + 1}" for i in range(len(df))]
    x = np.arange(len(df))
    w = 0.38

    ax.bar(x - w / 2, df["delta_true"], width=w, color=STYLE["true"], alpha=0.9, label="True Delta")
    ax.bar(
        x + w / 2,
        df["Pred_LogP_TVSTFN_delta"],
        width=w,
        color=STYLE["pred"],
        alpha=0.85,
        label="Pred Delta",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["pair"], fontsize=8)
    ax.set_ylabel("|Delta logPe|")
    ax.grid(axis="y", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8)


def panel_c(ax):
    p = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/wp2_stratified/stratified_metrics.csv")
    df = pd.read_csv(p)
    mw = df[df["stratifier"] == "MW"].copy()
    tp = df[df["stratifier"] == "TPSA"].copy()

    ax.plot(mw["bin"], mw["F1"], marker="o", markersize=4, lw=1.6, color=STYLE["mw"], label="MW bins")
    ax.plot(tp["bin"], tp["F1"], marker="s", markersize=4, lw=1.6, color=STYLE["tpsa"], label="TPSA bins")
    ax.set_ylim(0.45, 0.82)
    ax.set_ylabel("F1")
    ax.grid(alpha=0.7)
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    ax.legend(loc="upper right", fontsize=8)


def stamp_panel(ax, label: str):
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "#bfbfbf", "linewidth": 0.6, "pad": 2.0},
    )


def save_single_panels():
    for name, fn in [
        ("figure_3A_multicyc_style.png", panel_a),
        ("figure_3B_multicyc_style.png", panel_b),
        ("figure_3C_multicyc_style.png", panel_c),
    ]:
        fig, ax = plt.subplots(figsize=(7.8, 4.3), dpi=220)
        fn(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_group_figure() -> None:
    fig = plt.figure(figsize=(13.2, 9.4), dpi=240)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.06], hspace=0.23, wspace=0.16)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    stamp_panel(ax_a, "A")
    stamp_panel(ax_b, "B")
    stamp_panel(ax_c, "C")

    out_png = os.path.join(OUT, "figure_3_ABC_multicyc_style.png")
    out_tif = os.path.join(OUT, "figure_3_ABC_multicyc_style.tif")
    out_pdf = os.path.join(OUT, "figure_3_ABC_multicyc_style.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_tif, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


    cap_en = os.path.join(OUT, "figure_3_ABC_caption_multicyc_en.md")
    cap_zh = os.path.join(OUT, "figure_3_ABC_caption_multicyc_zh.md")
    with open(cap_en, "w", encoding="utf-8") as f:
        f.write(
            "Figure 3. MultiCycPermea-style redraw of TV-STFN section-3 evidence. "
            "(A) Overall metric comparison between tv5overnight_1 and hard_v1_overnight. "
            "(B) True versus predicted permeability deltas for top activity-cliff pairs. "
            "(C) Stratified F1 trends across MW and TPSA bins.\n"
        )
    with open(cap_zh, "w", encoding="utf-8") as f:
        f.write(
            "图3. 参考 MultiCycPermea 图形风格重绘的 TV-STFN 第3部分证据图。"
            "(A) tv5overnight_1 与 hard_v1_overnight 的总体指标对比；"
            "(B) 活性悬崖样本对的真实与预测差值对比；"
            "(C) 按 MW 与 TPSA 分层的 F1 趋势。\n"
        )


def main():
    set_global_style()
    save_single_panels()
    build_group_figure()
    print(os.path.join(OUT, "figure_3A_multicyc_style.png"))
    print(os.path.join(OUT, "figure_3B_multicyc_style.png"))
    print(os.path.join(OUT, "figure_3C_multicyc_style.png"))
    print(os.path.join(OUT, "figure_3_ABC_multicyc_style.png"))
    print(os.path.join(OUT, "figure_3_ABC_multicyc_style.tif"))
    print(os.path.join(OUT, "figure_3_ABC_multicyc_style.pdf"))
    print(os.path.join(OUT, "figure_3_ABC_caption_multicyc_en.md"))
    print(os.path.join(OUT, "figure_3_ABC_caption_multicyc_zh.md"))


if __name__ == "__main__":
    main()
