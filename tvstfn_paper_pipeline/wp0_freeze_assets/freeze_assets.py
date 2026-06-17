import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tvstfn_paper_pipeline.common.utils import ensure_dir


def copy_matches(src_dir: str, dst_dir: str, patterns):
    copied = []
    for pattern in patterns:
        for fp in glob.glob(os.path.join(src_dir, pattern)):
            if os.path.isfile(fp):
                target = os.path.join(dst_dir, os.path.basename(fp))
                shutil.copy2(fp, target)
                copied.append(target)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Freeze paper baseline assets for reproducible plotting")
    parser.add_argument("--pred-dir", type=str, default="benchmark_results")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/paper_v1")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    copied_files = copy_matches(
        args.pred_dir,
        args.out_dir,
        ["fold_*_predictions.csv", "fold_*_predictions_merged.csv", "*summary*.csv", "*per_fold*.csv"],
    )

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_prediction_dir": os.path.abspath(args.pred_dir),
        "output_dir": os.path.abspath(args.out_dir),
        "copied_file_count": len(copied_files),
        "copied_files": [os.path.basename(x) for x in copied_files],
        "notes": args.notes,
    }

    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Frozen assets saved to: {args.out_dir}")
    print(f"Copied files: {len(copied_files)}")


if __name__ == "__main__":
    main()
