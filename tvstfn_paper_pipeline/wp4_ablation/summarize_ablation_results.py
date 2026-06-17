import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/data/workplace/jwx/TV-STFN"
OUT_DIR = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/wp4_ablation")

VARIANTS = ["full", "wo_0d", "wo_1d", "wo_2d", "wo_3d"]


def pick_latest_file(pattern):
    cands = glob.glob(pattern)
    if not cands:
        return ""
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def load_rows(tag_prefix=""):
    rows = []
    for variant in VARIANTS:
        if tag_prefix:
            p = pick_latest_file(os.path.join(OUT_DIR, f"ablation_summary_{tag_prefix}_{variant}_gpu*.csv"))
        else:
            p = pick_latest_file(os.path.join(OUT_DIR, f"ablation_summary_{variant}_gpu*.csv"))
            if not p:
                # fallback to latest strat run shard if present
                p = pick_latest_file(os.path.join(OUT_DIR, f"ablation_summary_strat*_{variant}_gpu*.csv"))
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        rows.append(
            {
                "variant": variant,
                "ACC_mean": float(r["ACC_mean"]),
                "ACC_std": float(r["ACC_std"]),
                "F1_mean": float(r["F1_mean"]),
                "F1_std": float(r["F1_std"]),
                "AUROC_mean": float(r["AUROC_mean"]),
                "AUROC_std": float(r["AUROC_std"]),
            }
        )
    return pd.DataFrame(rows)


def add_deltas(df):
    if "full" not in set(df["variant"]):
        return df
    full = df[df["variant"] == "full"].iloc[0]
    for m in ["ACC_mean", "F1_mean", "AUROC_mean"]:
        df[f"delta_vs_full_{m.replace('_mean','')}"] = df[m] - float(full[m])
    return df


def save_plot(df, suffix=""):
    order = ["full", "wo_0d", "wo_1d", "wo_2d", "wo_3d"]
    d = df.set_index("variant").reindex(order).dropna(how="all").reset_index()

    x = range(len(d))
    width = 0.25

    plt.figure(figsize=(10.5, 5.2), dpi=180)
    plt.bar([i - width for i in x], d["ACC_mean"], width=width, label="ACC", color="#4C78A8")
    plt.bar([i for i in x], d["F1_mean"], width=width, label="F1", color="#F58518")
    plt.bar([i + width for i in x], d["AUROC_mean"], width=width, label="AUROC", color="#54A24B")
    plt.xticks(list(x), d["variant"])
    plt.ylim(0.0, 0.9)
    plt.ylabel("Score")
    plt.title("WP4 Ablation Summary (TV-STFN)")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    name = f"ablation_summary_complete{suffix}_bar.png"
    plt.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    plt.close()


def save_markdown(df, suffix=""):
    md_path = os.path.join(OUT_DIR, f"ablation_summary_complete{suffix}.md")
    d = df.copy()
    cols = [
        "variant",
        "ACC_mean",
        "F1_mean",
        "AUROC_mean",
        "delta_vs_full_ACC",
        "delta_vs_full_F1",
        "delta_vs_full_AUROC",
    ]
    for c in cols[1:]:
        d[c] = d[c].map(lambda x: f"{x:.4f}")

    lines = []
    lines.append("# WP4 Ablation Complete Summary")
    lines.append("")
    lines.append("| variant | ACC_mean | F1_mean | AUROC_mean | ΔACC vs full | ΔF1 vs full | ΔAUROC vs full |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in d[cols].iterrows():
        lines.append(
            f"| {r['variant']} | {r['ACC_mean']} | {r['F1_mean']} | {r['AUROC_mean']} | {r['delta_vs_full_ACC']} | {r['delta_vs_full_F1']} | {r['delta_vs_full_AUROC']} |"
        )

    lines.append("")
    lines.append(f"Figure: ablation_summary_complete{suffix}_bar.png")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize ablation results")
    parser.add_argument("--tag-prefix", type=str, default="", help="Prefer files of this run prefix, e.g. stratv3")
    args = parser.parse_args()

    df = load_rows(tag_prefix=args.tag_prefix)
    if df.empty:
        raise RuntimeError("No ablation summary files found")
    df = add_deltas(df)

    suffix = f"_{args.tag_prefix}" if args.tag_prefix else ""
    csv_out = os.path.join(OUT_DIR, f"ablation_summary_complete{suffix}.csv")
    df.to_csv(csv_out, index=False)

    save_plot(df, suffix=suffix)
    save_markdown(df, suffix=suffix)

    print(csv_out)
    print(os.path.join(OUT_DIR, f"ablation_summary_complete{suffix}.md"))
    print(os.path.join(OUT_DIR, f"ablation_summary_complete{suffix}_bar.png"))


if __name__ == "__main__":
    main()
