import argparse
import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_reducer(x, n_neighbors=25, min_dist=0.15, random_state=42):
    try:
        umap = importlib.import_module("umap")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        z = reducer.fit_transform(x)
        method = "UMAP"
    except Exception:
        TSNE = importlib.import_module("sklearn.manifold").TSNE

        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
        z = reducer.fit_transform(x)
        method = "TSNE"
    return z, method


def main():
    parser = argparse.ArgumentParser(description="Plot dual-view UMAP for 0D vs fused TV-STFN features")
    parser.add_argument("--npz", type=str, default="tvstfn_paper_pipeline/outputs/wp1_umap/umap_embeddings.npz")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp1_umap")
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--max-points", type=int, default=2500, help="cap points for reducer speed; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.npz)
    x_fuse = data["feat_fusion"]
    x_0d = data["feat_0d"]
    y_true = data["y_true"]
    y_bin = (y_true >= -6.0).astype(int)

    if args.max_points > 0 and len(y_true) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(y_true), size=args.max_points, replace=False))
        x_fuse = x_fuse[keep]
        x_0d = x_0d[keep]
        y_true = y_true[keep]
        y_bin = y_bin[keep]

    z_0d, method_0d = run_reducer(x_0d, args.n_neighbors, args.min_dist, random_state=args.seed)
    z_fuse, method_fuse = run_reducer(x_fuse, args.n_neighbors, args.min_dist, random_state=args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    for ax, z, title in [
        (axes[0], z_0d, f"{method_0d}: Raw 0D Features"),
        (axes[1], z_fuse, f"{method_fuse}: TV-STFN Fused Features"),
    ]:
        sc = ax.scatter(z[:, 0], z[:, 1], c=y_bin, cmap="coolwarm", s=10, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.04, label="Label (0=Low, 1=High)")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "figure_Y_umap_dual.png")
    plt.savefig(fig_path, bbox_inches="tight")

    try:
        silhouette_score = importlib.import_module("sklearn.metrics").silhouette_score

        metrics = {
            "silhouette_0d": float(silhouette_score(z_0d, y_bin)) if len(np.unique(y_bin)) > 1 else np.nan,
            "silhouette_fused": float(silhouette_score(z_fuse, y_bin)) if len(np.unique(y_bin)) > 1 else np.nan,
        }
    except Exception:
        metrics = {"silhouette_0d": np.nan, "silhouette_fused": np.nan}
    pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, "umap_quality_metrics.csv"), index=False)

    coord = pd.DataFrame(
        {
            "idx": np.arange(len(y_true)),
            "y_true": y_true,
            "y_bin": y_bin,
            "x0d_1": z_0d[:, 0],
            "x0d_2": z_0d[:, 1],
            "xfuse_1": z_fuse[:, 0],
            "xfuse_2": z_fuse[:, 1],
        }
    )
    coord.to_csv(os.path.join(args.out_dir, "umap_coordinates.csv"), index=False)

    print(f"Saved plot: {fig_path}")
    print(f"Silhouette 0D={metrics['silhouette_0d']:.4f}, Fused={metrics['silhouette_fused']:.4f}")


if __name__ == "__main__":
    main()
