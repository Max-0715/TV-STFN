import argparse
import os
import subprocess
import time
from collections import deque


def query_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows = []
    for line in out.strip().splitlines():
        i, util, mem_used, mem_total = [x.strip() for x in line.split(",")]
        rows.append(
            {
                "gpu": int(i),
                "util": int(float(util)),
                "mem_used": int(float(mem_used)),
                "mem_total": int(float(mem_total)),
            }
        )
    return rows


def is_gpu_free(info, util_thr, mem_thr):
    return info["util"] <= util_thr and info["mem_used"] <= mem_thr


def spawn_task(py, root, out_dir, gpu, variant, epochs, batch_size, n_folds, seed, lr, lambda_cls, lambda_entropy, tag_prefix):
    if tag_prefix:
        tag = f"{tag_prefix}_{variant}_gpu{gpu}"
    else:
        tag = f"{variant}_gpu{gpu}"
    log = os.path.join(out_dir, f"{tag}.log")
    cmd = [
        py,
        "tvstfn_paper_pipeline/wp4_ablation/run_ablation_cv.py",
        "--data-dir",
        "tetraview_processed",
        "--out-dir",
        out_dir,
        "--n-folds",
        str(n_folds),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--lambda-cls",
        str(lambda_cls),
        "--lambda-entropy",
        str(lambda_entropy),
        "--seed",
        str(seed),
        "--variants",
        variant,
        "--tag",
        tag,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    f = open(log, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p, log, tag


def main():
    parser = argparse.ArgumentParser(description="Dispatch ablation variants to all currently free GPUs in real time")
    parser.add_argument("--root", type=str, default="/data/workplace/jwx/TV-STFN")
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp4_ablation")
    parser.add_argument("--variants", type=str, default="full,wo_0d,wo_1d,wo_2d,wo_3d")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cls", type=float, default=0.3)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag-prefix", type=str, default="")
    parser.add_argument("--util-thr", type=int, default=10)
    parser.add_argument("--mem-thr", type=int, default=1200)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    queue = deque([v.strip() for v in args.variants.split(",") if v.strip()])
    running = {}

    print(f"[dispatcher] pending variants: {list(queue)}", flush=True)
    while queue or running:
        # collect finished
        done = []
        for gpu, rec in running.items():
            p = rec["proc"]
            code = p.poll()
            if code is not None:
                done.append((gpu, code, rec["tag"], rec["log"]))
        for gpu, code, tag, log in done:
            running.pop(gpu, None)
            print(f"[done] gpu={gpu} tag={tag} exit={code} log={log}", flush=True)
            if code != 0:
                raise RuntimeError(f"Task failed: {tag}, see log {log}")

        # schedule
        if queue:
            gpu_infos = query_gpus()
            free_gpus = [g["gpu"] for g in gpu_infos if is_gpu_free(g, args.util_thr, args.mem_thr)]
            print(
                f"[tick] pending={len(queue)} running={len(running)} free_gpus={free_gpus}",
                flush=True,
            )
            for gpu in free_gpus:
                if not queue:
                    break
                if gpu in running:
                    continue
                variant = queue.popleft()
                proc, log, tag = spawn_task(
                    args.python_bin,
                    root,
                    out_dir,
                    gpu,
                    variant,
                    args.epochs,
                    args.batch_size,
                    args.n_folds,
                    args.seed,
                    args.lr,
                    args.lambda_cls,
                    args.lambda_entropy,
                    args.tag_prefix,
                )
                running[gpu] = {"proc": proc, "tag": tag, "log": log}
                print(f"[start] gpu={gpu} variant={variant} tag={tag}", flush=True)

        if queue or running:
            time.sleep(args.poll_seconds)

    # merge after all shards complete
    merge_cmd = [
        args.python_bin,
        "tvstfn_paper_pipeline/wp4_ablation/merge_ablation_parts.py",
        "--out-dir",
        out_dir,
        "--tag-prefix",
        args.tag_prefix,
    ]
    subprocess.check_call(merge_cmd, cwd=root)
    print("[dispatcher] all variants finished and merged.", flush=True)


if __name__ == "__main__":
    main()
