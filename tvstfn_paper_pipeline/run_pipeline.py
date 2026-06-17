import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class Step:
    name: str
    cmd: List[str]
    outputs: List[str]
    expensive: bool = False


def exists_all(paths: List[str]) -> bool:
    return all(os.path.exists(p) for p in paths)


def run_step(step: Step, force: bool, resume: bool) -> None:
    if resume and not force and exists_all(step.outputs):
        print(f"[skip] {step.name}: outputs already exist")
        return

    print(f"[run ] {step.name}")
    proc = subprocess.run(step.cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {step.name}")

    missing = [p for p in step.outputs if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Step finished but missing outputs in {step.name}: {missing}")
    print(f"[ ok ] {step.name}")


def build_steps(py: str, mode: str) -> List[Step]:
    steps = [
        Step(
            "wp0_freeze_assets",
            [
                py,
                "tvstfn_paper_pipeline/wp0_freeze_assets/freeze_assets.py",
                "--pred-dir",
                "benchmark_results",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/paper_v1",
            ],
            ["tvstfn_paper_pipeline/outputs/paper_v1/metadata.json"],
        ),
        Step(
            "wp1_export_embeddings",
            [
                py,
                "tvstfn_paper_pipeline/wp1_umap/export_embeddings.py",
                "--data-dir",
                "tetraview_processed",
                "--weights",
                "best_tetraview_model.pth",
                "--pred-dir",
                "benchmark_results",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/wp1_umap",
                "--max-samples",
                "2500",
            ],
            ["tvstfn_paper_pipeline/outputs/wp1_umap/umap_embeddings.npz"],
            expensive=True,
        ),
        Step(
            "wp1_plot_umap",
            [
                py,
                "tvstfn_paper_pipeline/wp1_umap/plot_umap.py",
                "--npz",
                "tvstfn_paper_pipeline/outputs/wp1_umap/umap_embeddings.npz",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/wp1_umap",
                "--max-points",
                "2500",
            ],
            ["tvstfn_paper_pipeline/outputs/wp1_umap/figure_Y_umap_dual.png"],
        ),
        Step(
            "wp2_stratified",
            [
                py,
                "tvstfn_paper_pipeline/wp2_stratified_robustness/stratified_eval.py",
                "--pred-dir",
                "benchmark_results",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/wp2_stratified",
            ],
            ["tvstfn_paper_pipeline/outputs/wp2_stratified/stratified_metrics.csv"],
        ),
        Step(
            "wp3_find_cliffs",
            [
                py,
                "tvstfn_paper_pipeline/wp3_activity_cliff/find_activity_cliffs.py",
                "--pred-dir",
                "benchmark_results",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/wp3_cliff",
            ],
            ["tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv"],
        ),
        Step(
            "wp3_export_attention",
            [
                py,
                "tvstfn_paper_pipeline/wp3_activity_cliff/export_conformer_attention.py",
                "--indices-csv",
                "tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv",
                "--data-dir",
                "tetraview_processed",
                "--weights",
                "best_tetraview_model.pth",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/wp3_cliff",
                "--smiles-csv",
                "CycPeptMPDB_Peptide_PAMPA.csv",
            ],
            ["tvstfn_paper_pipeline/outputs/wp3_cliff/conformer_attention_top.csv"],
            expensive=True,
        ),
        Step(
            "wp3_build_figure_data",
            [
                py,
                "tvstfn_paper_pipeline/wp3_activity_cliff/build_cliff_figure_data.py",
                "--cliff-csv",
                "tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv",
                "--attn-csv",
                "tvstfn_paper_pipeline/outputs/wp3_cliff/conformer_attention_top.csv",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/wp3_cliff",
            ],
            ["tvstfn_paper_pipeline/outputs/wp3_cliff/figure_Z_panel_data.csv"],
        ),
        Step(
            "wp5_stats_calibration",
            [
                py,
                "tvstfn_paper_pipeline/wp5_stats_calibration/stats_and_calibration.py",
                "--pred-dir",
                "benchmark_results",
                "--out-dir",
                "tvstfn_paper_pipeline/outputs/wp5_stats",
                "--focus-model",
                "TVSTFN",
            ],
            ["tvstfn_paper_pipeline/outputs/wp5_stats/stats_summary.csv"],
        ),
    ]

    if mode == "full":
        steps.extend(
            [
                Step(
                    "wp4_ablation_run",
                    [
                        py,
                        "tvstfn_paper_pipeline/wp4_ablation/dispatch_ablation_free_gpus.py",
                        "--root",
                        ".",
                        "--python-bin",
                        py,
                        "--out-dir",
                        "tvstfn_paper_pipeline/outputs/wp4_ablation",
                        "--n-folds",
                        "5",
                        "--epochs",
                        "25",
                    ],
                    ["tvstfn_paper_pipeline/outputs/wp4_ablation/ablation_summary.csv"],
                    expensive=True,
                ),
                Step(
                    "wp4_ablation_plot",
                    [
                        py,
                        "tvstfn_paper_pipeline/wp4_ablation/plot_ablation.py",
                        "--summary-csv",
                        "tvstfn_paper_pipeline/outputs/wp4_ablation/ablation_summary.csv",
                        "--out-dir",
                        "tvstfn_paper_pipeline/outputs/wp4_ablation",
                    ],
                    ["tvstfn_paper_pipeline/outputs/wp4_ablation/figure_X_ablation.png"],
                ),
            ]
        )

    return steps


def main():
    parser = argparse.ArgumentParser(description="Smart paper pipeline runner with resume/skip")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="skip completed steps based on output files")
    parser.add_argument("--force", action="store_true", help="rerun selected steps even when outputs exist")
    parser.add_argument("--only", default="", help="comma-separated step names to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["PYTHONPATH"] = os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")

    steps = build_steps(args.python_bin, args.mode)
    if args.only.strip():
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        steps = [s for s in steps if s.name in allow]

    if args.dry_run:
        for s in steps:
            print(s.name, "=>", " ".join(s.cmd))
        return

    for step in steps:
        run_step(step, force=args.force, resume=args.resume)

    print("All requested steps completed.")


if __name__ == "__main__":
    main()
