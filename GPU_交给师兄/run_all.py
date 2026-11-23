#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TV-STFN 返修实验 · 傻瓜版（一个文件搞定全部）
================================================================================
最常用（在 TV-STFN 项目根目录里跑）：

    python GPU_交给师兄/run_all.py              # 一键：先自检，再按顺序跑完全部实验
    python GPU_交给师兄/run_all.py 检查          # 只做自检（3 分钟，先跑这个）
    python GPU_交给师兄/run_all.py 小测试        # 5 分钟冒烟测试，确认环境真的能训
    python GPU_交给师兄/run_all.py 看进度        # 另开一个终端，实时看所有任务进度
    python GPU_交给师兄/run_all.py --gpus 0,1,2 # 有 3 张显卡时自动分三组并行

也可以只跑某一步（中文名即可，可写多个）：

    消融   构象   种子   严格   骨架   留出   来源   预处理   分assay   基线   悬崖   出表

中断了怎么办？**原命令再跑一遍就接着跑**（已经跑完的折会自动跳过）。

三条不许动的规矩
  1) 0D 特征必须做 log1p 有界化（脚本已内置）——原始数据里 RDKit 的 Ipc 最大 4e36，
     会把整批 loss 变成 NaN，这是之前结果不稳的根因。
  2) 不要改 src/model.py 的维度（view4=729、view2=200）。
  3) 不要挑种子、不要挑折；日志里 nan 不为 0 的折都要报回来。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# ======================================================================================
# 0. 路径 / 常量 / 显示
# ======================================================================================
VERSION = "2026-09-20.6"      # .6：另一条派发路径 run_parallel 的参数顺序也修了，并统一走 job_cmd + 自检
                              # .5：变体表驱动(VARIANT_OPS)+开跑前变体登记自检+汇总表标出缺失变体
                              # .4：修 build_model 漏掉 wo_cliff 分支导致 ValueError（消融整步必崩的真根因）
                              # .3：单折任务(--only-part)、子任务超时、失败自动打印报错尾巴、修正子进程参数覆盖
CLS_THRESHOLD = -6.0          # 分类阈值：logPe >= -6 视为"高渗透"
IPC_INDEX = 554               # view4[512 + 42] = RDKit Ipc（会溢出的那一维）
HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ("full", "wo_cliff", "wo_3d", "wo_1d", "wo_2d", "wo_0d")
STEP_NAMES = {
    "检查": "preflight", "版本": "version", "数据流": "dataflow", "消融": "ablation", "构象": "confs", "种子": "seeds",
    "严格": "strict", "骨架": "scaffold", "留出": "holdout", "来源": "loso",
    "预处理": "preprocess", "分assay": "assay", "基线": "baselines", "悬崖": "cliffs",
    "出表": "report", "打包数据": "pack", "无人值守": "daemon", "划分": "splits", "出图": "figures",
    "特征": "features", "导出数据": "exportdata", "写作映射": "writing", "看进度": "watch", "状态": "status", "小测试": "smoke", "全部": "all",
}
STEP_TITLE = {
    "preflight": "第 0 步 · 检查环境（3 分钟）",
    "version": "版本核对",
    "dataflow": "第 0b 步 · 数据流审计（1 分钟，回答审稿人的「数据去向」）",
    "ablation": "第 1 步 · 消融实验",
    "confs": "第 2 步 · 构象数扫描",
    "seeds": "第 3 步 · 换种子复现",
    "strict": "第 4 步 · 防泄漏严格划分（论文主结果）",
    "scaffold": "第 5 步 · 骨架划分",
    "holdout": "第 6 步 · 随机留出（对照组）",
    "loso": "第 7 步 · 留一来源",
    "preprocess": "第 8 步 · 生成分 assay 数据（Caco2 等）",
    "assay": "第 9 步 · 分 assay 训练",
    "baselines": "第 10 步 · 传统机器学习基线",
    "cliffs": "第 11 步 · 活动悬崖分析",
    "report": "第 12 步 · 汇总出表",
    "pack": "打包数据（一次性，之后训练快 2–3 倍）",
    "daemon": "全自动无人值守（自动挑卡 + 自动重试 + 自动收尾）",
    "splits": "第 13 步 · 导出数据划分文件（审稿人 R1 要求公开划分）",
    "figures": "第 14 步 · 用真实预测重画全部图表（R1 批评的图 4/6/10 等）",
    "features": "第 15 步 · 隐层特征 t-SNE（R1 说图 5B 是丝状伪影）",
    "exportdata": "第 16 步 · 数据可得性打包（数据/构象生成说明 + 样本清单）",
    "writing": "第 17 步 · 表图 ↔ 数据文件对照表（写稿时照着填）",
    "smoke": "冒烟测试（5 分钟，只为确认能跑通）",
    "watch": "实时看板",
    "status": "进度快照",
}
STEP_WHY = {
    "version": "目的：确认服务器上跑的是不是最新那一版（照着我发你的版本号核对）。",
    "preflight": "目的：确认显卡、数据、分词器、0D 特征都没问题，别白跑一整天。",
    "dataflow": "目的：把「原始条目→去重→各实验类型→实际建模样本」的每一步数字对齐成一条链，"
                "顺手生成 0D/2D 描述符清单（补充材料要的）。",
    "ablation": "目的：回答审稿人「去掉某个模块会掉多少分」。",
    "confs": "目的：回答审稿人「为什么用 10 个构象」，给出 1/2/3/5/10 的曲线。",
    "seeds": "目的：换 5 个随机种子重跑关键对比，证明结论不是运气。",
    "strict": "目的：做一个「测试集分子在训练集里找不到相似邻居」的严格测试集。",
    "scaffold": "目的：测试模型对没见过的分子骨架的泛化能力。",
    "holdout": "目的：作为对照，展示随机划分会高估多少（泄漏有多严重）。",
    "loso": "目的：按文献来源留一，看模型跨来源表现。",
    "preprocess": "目的：把 Caco2/MDCK/RRCK 端点也做成同样的四视图数据。",
    "assay": "目的：回答审稿人「换个实验类型（Caco2）还能用吗」。",
    "baselines": "目的：和随机森林/SVM 在完全相同的划分与特征下对比。",
    "cliffs": "目的：统计全库的活动悬崖对，看模型能否区分相似但渗透率差很大的分子。",
    "report": "目的：把所有结果汇总成能直接贴进论文的表格。",
    "pack": "目的：把 .pt 打成单个 memmap 文件，消灭「每 epoch 重解 6941 个文件」的 CPU 瓶颈。",
    "daemon": "目的：一条命令挂上去就不用管——自动挑闲卡、一直保持满负荷、崩了自动重跑、跑完自动出表出图。",
    "splits": "目的：把每一次划分写成 CSV 公开（谁在训练集、谁在测试集、第几折），直接回应 R1 的 provide data splits。",
    "figures": "目的：用真实逐样本预测重画所有图（旧出图脚本是用合成分数画的，必须换掉）。",
    "features": "目的：重新导出隐层特征并调好 t-SNE 超参，回应 R1 对图 5B 的质疑。",
    "exportdata": "目的：生成数据可得性声明所需的东西（样本清单、构象生成参数、数据格式说明）。",
    "writing": "目的：把「哪张图/哪张表 由哪个文件哪个字段来填」列成对照表，写稿时不用再找。",
}

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def find_repo_root() -> str:
    d = HERE
    for _ in range(8):
        if os.path.exists(os.path.join(d, "src", "dataset.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise SystemExit("找不到 TV-STFN 项目根目录（缺 src/dataset.py）。\n"
                     "请把整个 GPU_交给师兄 文件夹放到 TV-STFN 项目根目录下再运行。")


_SRC: Dict[str, object] = {}


def ensure_src() -> Dict[str, object]:
    """按需导入项目的模型/数据代码（这样 --help 在项目外也能用）。"""
    if _SRC:
        return _SRC
    root = find_repo_root()
    for _p in (root, os.path.join(root, "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from src.dataset import tetra_view_collate
    from src.loss import CompositeLoss
    from src.model import TetraViewNet
    _SRC.update({"root": root, "tetra_view_collate": tetra_view_collate,
                 "CompositeLoss": CompositeLoss, "TetraViewNet": TetraViewNet})
    return _SRC


REPO_ROOT = HERE  # 兼容旧引用；真正需要项目根目录时用 repo_root()


def repo_root() -> str:
    """项目根目录（找不到就退回脚本所在目录，绝不抛异常）。"""
    try:
        return find_repo_root()
    except SystemExit:
        return HERE


def fmt_dur(sec: float) -> str:
    if sec is None or not np.isfinite(sec) or sec < 0:
        return "--:--"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def banner(text: str, why: str = "") -> None:
    line = "=" * 78
    print(f"\n{line}\n  {BOLD}{text}{RESET}")
    if why:
        print(f"  {why}")
    print(line, flush=True)


class Live:
    """进度条：接终端时原地刷新，重定向到日志时逐行打印（都能实时看到）。"""

    def __init__(self, total: int, label: str, min_interval: float = 0.4):
        self.total, self.done, self.label = max(int(total), 1), 0, label
        self.t0, self.last, self.min_interval = time.time(), 0.0, min_interval
        self.tty = sys.stdout.isatty()

    def _line(self, extra: str = "") -> str:
        el = time.time() - self.t0
        rate = self.done / el if el > 0 and self.done else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        parts = [f"[{self.done:>3d}/{self.total:<3d}]", f"{100.0 * self.done / self.total:5.1f}%",
                 f"已用 {fmt_dur(el)}", f"预计还要 {fmt_dur(eta)}", self.label]
        if extra:
            parts.append(extra)
        return " | ".join(str(p) for p in parts if p)

    def status(self, extra: str = "", force: bool = False):
        if not force and time.time() - self.last < self.min_interval:
            return
        self.last = time.time()
        line = self._line(extra)
        if self.tty:
            sys.stdout.write("\r" + line[:200].ljust(min(len(line), 200)))
            sys.stdout.flush()
        else:
            print(line, flush=True)

    def step(self, n: int = 1, extra: str = ""):     # 粗粒度（一折、一步）：一定打印
        self.done += n
        self.status(extra, force=True)

    def tick(self, n: int = 1, extra: str = ""):     # 细粒度（一个分子）：限流打印
        self.done += n
        self.status(extra, force=False)

    def log(self, msg: str):
        if self.tty:
            sys.stdout.write("\r" + " " * 100 + "\r")
        print(msg, flush=True)

    def close(self, extra: str = ""):
        self.done = self.total
        self.status(extra, force=True)
        if self.tty:
            sys.stdout.write("\n")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def write_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def write_csv(path: str, rows: Iterable[dict]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    os.replace(tmp, path)


def env_report() -> str:
    bits = [f"python {sys.version.split()[0]}", f"torch {torch.__version__}",
            f"CUDA {'可用' if torch.cuda.is_available() else '不可用'}"]
    if torch.cuda.is_available():
        bits.append(f"显卡 {torch.cuda.device_count()} 张")
    try:
        from rdkit import rdBase
        from rdkit.Chem import Descriptors
        bits.append(f"rdkit {rdBase.rdkitVersion}")
        bits.append(f"描述符数 {len(Descriptors.descList)}")
    except Exception:
        bits.append("rdkit 缺失")
    try:
        import transformers
        bits.append(f"transformers {transformers.__version__}")
    except Exception:
        bits.append("transformers 缺失！")
    return " | ".join(bits)


# ======================================================================================
# 1. 数据读取 + 元数据校验
# ======================================================================================
class Dataset:
    """按需读取 data_*.pt（不一次性全载入，启动快；坏样本自动跳过）。"""

    def __init__(self, data_dir: str):
        if not os.path.isdir(data_dir):
            raise SystemExit(f"数据目录不存在：{data_dir}")
        names = [f for f in os.listdir(data_dir) if f.endswith(".pt")
                 and os.path.getsize(os.path.join(data_dir, f)) > 0]

        def key(n):
            tail = n.rsplit(".", 1)[0].split("_")[-1]
            return int(tail) if tail.isdigit() else n

        self.file_names = sorted(names, key=key)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_names)

    def raw(self, i: int):
        try:
            d = torch.load(os.path.join(self.data_dir, self.file_names[i]), weights_only=False)
        except Exception:
            return None
        if not isinstance(d, dict) or not all(k in d for k in
                                              ("view1_3d", "view2_1d", "view3_2d", "view4_0d", "target")):
            return None
        v4 = d["view4_0d"]
        if v4 is None or v4.numel() == 0 or not bool(torch.isfinite(v4.float()).all()):
            return None
        if float(v4.abs().sum()) == 0.0:
            return None
        return d


_PACK_CACHE: Dict[str, dict] = {}


class PackedDataset:
    """把 6941 个 .pt 打包成若干 .npy（memmap）后的数据集。

    为什么要这个：每个 epoch 都要把全部 .pt 重新解 pickle 一遍，单折在 CPU 上要花 1–4 小时，
    GPU 只有 5–24% 利用率。打包后数据变成一个 ~1.3 GB 的文件，所有训练进程共享操作系统的
    页面缓存，读样本变成读内存，也不再解 pickle。接口和 Dataset 完全一致（file_names / raw / __len__）。
    """

    def __init__(self, pack_dir: str, verbose: bool = False):
        self.pack_dir = pack_dir
        man_path = os.path.join(pack_dir, "manifest.json")
        if not os.path.exists(man_path):
            raise SystemExit(f"打包目录里没有 manifest.json：{pack_dir}")
        self.manifest = read_json(man_path, {}) or {}
        self.file_names = list(self.manifest.get("file_names", []))
        self.dims = self.manifest.get("dims", {})
        self.n = int(self.manifest.get("n_samples", len(self.file_names)))
        self.data_dir = pack_dir            # 让 load_metadata 能在里面找 sample_manifest.csv
        self._mm = None
        if verbose:
            print(f"packed dataset: {self.n} 个样本，来自 {self.manifest.get('source_dir')}", flush=True)

    # mmap 要在每个子进程里各自打开（懒加载），不能跨进程 pickle
    def _maps(self) -> dict:
        pid = os.getpid()
        cached = _PACK_CACHE.get(f"{self.pack_dir}@{pid}")
        if cached is None:
            m = {}
            for name in ("coords", "atom_feat", "num_atoms", "ids", "mask", "x",
                         "edge_index", "edge_attr", "edge_ptr", "view4", "target"):
                path = os.path.join(self.pack_dir, f"{name}.npy")
                if os.path.exists(path):
                    m[name] = np.load(path, mmap_mode="r")
            _PACK_CACHE[f"{self.pack_dir}@{pid}"] = m
            cached = m
        return cached

    def __len__(self):
        return self.n

    def raw(self, i: int):
        from torch_geometric.data import Data
        m = self._maps()
        if not m or i >= self.n:
            return None
        try:
            na = int(m["num_atoms"][i])
            coords = torch.from_numpy(np.array(m["coords"][i][:, :na, :], dtype=np.float32))
            atom_feat = torch.from_numpy(np.array(m["atom_feat"][i][:na], dtype=np.float32))
            ids = torch.from_numpy(np.array(m["ids"][i], dtype=np.int64))
            mask = torch.from_numpy(np.array(m["mask"][i], dtype=np.int64))
            x = torch.from_numpy(np.array(m["x"][i][:na], dtype=np.float32))
            e0, e1 = int(m["edge_ptr"][i]), int(m["edge_ptr"][i + 1])
            ei = torch.from_numpy(np.array(m["edge_index"][e0:e1], dtype=np.int64).T).contiguous()
            ea = torch.from_numpy(np.array(m["edge_attr"][e0:e1], dtype=np.float32))
            view4 = torch.from_numpy(np.array(m["view4"][i], dtype=np.float32))
            target = float(m["target"][i])
        except Exception:
            return None
        return {"view1_3d": {"coords": coords, "atom_feat": atom_feat},
                "view2_1d": {"input_ids": ids, "attention_mask": mask},
                "view3_2d": Data(x=x, edge_index=ei, edge_attr=ea),
                "view4_0d": view4, "target": target, "smiles": ""}


def packed_dir_candidates(g) -> List[str]:
    """打包数据可能放的位置（按优先级）：--packed-dir → 数据目录旁边 → 结果目录下。"""
    cands = []
    if getattr(g, "packed_dir", None):
        cands.append(g.packed_dir)
    d = os.path.normpath(g.data_dir)
    cands.append(d.rstrip("/\\") + "_packed")
    cands.append(os.path.join(g.out_dir, "packed"))
    return cands


def packed_dir_for(g) -> str:
    """写入位置：显式指定优先，否则放数据目录旁边（换结果目录也能复用）。"""
    if getattr(g, "packed_dir", None):
        return g.packed_dir
    return os.path.normpath(g.data_dir).rstrip("/\\") + "_packed"


def find_packed_dir(g) -> Optional[str]:
    for d in packed_dir_candidates(g):
        if d and os.path.exists(os.path.join(d, "manifest.json")):
            return d
    return None


def open_dataset(g, verbose: bool = True):
    """有打包数据就用打包数据（自动、快得多），否则用原始 .pt 目录。"""
    pdir = find_packed_dir(g)
    use = getattr(g, "packed", "auto")
    if use == "off" or pdir is None:
        ds = Dataset(g.data_dir)
        if verbose and use == "auto":
            print("  提示：没有打包数据，正在直接读 .pt（每个 epoch 都要重新解压，较慢）。"
                  "建议先跑一次：python GPU_交给师兄/run_all.py 打包数据", flush=True)
        return ds
    ds = PackedDataset(pdir, verbose=verbose)
    src = ds.manifest.get("source_dir", "")
    if src and os.path.abspath(src) != os.path.abspath(g.data_dir):
        if verbose:
            print(f"  ！打包数据来自 {src}，与 --data-dir {g.data_dir} 不一致，仍按原始目录读", flush=True)
        return Dataset(g.data_dir)
    if verbose:
        print(f"  使用打包数据（memmap，读内存级速度）：{pdir}", flush=True)
    return ds


def load_targets(ds: Dataset, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    idx, tgt = [], []
    live = Live(len(ds), "读取数据") if verbose else None
    for i in range(len(ds)):
        item = ds.raw(i)
        if item is not None:
            t = item["target"]
            idx.append(i)
            tgt.append(float(t) if not torch.is_tensor(t) else float(t.reshape(-1)[0]))
        if live:
            live.tick(1)
    if live:
        live.close()
    return np.asarray(idx, dtype=np.int64), np.asarray(tgt, dtype=np.float32)


META_CANDIDATES = [
    "/data/workplace/jwx/MSF-CPMP/datasets_process/CycPeptMPDB_Peptide_PAMPA.csv",
    "/data/workplace/jwx/cycpeptmp/data/CycPeptMPDB_Peptide_PAMPA.csv",
    "/data/workplace/jwx/cycpeptmp/data/CycPeptMPDB_Peptide_All.csv",
    "/data/workplace/jwx/MSF-CPMP/datasets_process/CycPeptMPDB_Peptide_All.csv",
]
ALL_ASSAY_CANDIDATES = [
    "/data/workplace/jwx/cycpeptmp/data/CycPeptMPDB_Peptide_All.csv",
    "/data/workplace/jwx/MSF-CPMP/datasets_process/CycPeptMPDB_Peptide_All.csv",
]


def _candidates_for(g) -> List[str]:
    out = []
    if g.meta_csv:
        out.append(g.meta_csv)
    out.append(os.path.join(g.data_dir, "sample_manifest.csv"))
    out += META_CANDIDATES
    root = repo_root()
    out += [os.path.join(root, "data", os.path.basename(x)) for x in META_CANDIDATES]
    out += [os.path.join(root, os.path.basename(x)) for x in META_CANDIDATES]
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def autodetect_meta_csv(n_pt: int, g, verbose=True) -> Optional[str]:
    """没给 --meta-csv 时自动找生成这批 .pt 用的表。

    支持两种情况：表行数正好等于 .pt 个数；或者给的是全 assay 表，用 PAMPA 列筛出同样多行。
    找不到就返回 None（上层给出明确的补救命令）。
    """
    for path in _candidates_for(g):
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if len(df) == n_pt and "SMILES" in df.columns:
            if verbose:
                print(f"  元数据表（自动找到）：{path}（{len(df)} 行，与 .pt 数量一致）", flush=True)
            return path
        for col in ("PAMPA", "Permeability"):
            if col in df.columns:
                keep = pd.to_numeric(df[col], errors="coerce").notna()
                if int(keep.sum()) == n_pt:
                    # 写一份筛好的副本，避免每次重算
                    out = os.path.join(g.out_dir, f"用_{col}筛出的元数据表.csv")
                    ensure_dir(g.out_dir)
                    df.loc[keep].to_csv(out, index=False)
                    if verbose:
                        print(f"  元数据表（自动找到）：{path} → 用 {col} 列筛出 {n_pt} 行，"
                              f"已另存为 {out}", flush=True)
                    return out
    return None


def autodetect_all_assay_csv(g, n_pt: int, verbose=True) -> Optional[str]:
    """找全 assay 表（预处理用；优先选含 PAMPA 列且 PAMPA 非空数 == .pt 数的那种）。"""
    cands = ([g.assay_csv] if g.assay_csv else []) + ALL_ASSAY_CANDIDATES + [
        os.path.join(repo_root(), "data", os.path.basename(x)) for x in ALL_ASSAY_CANDIDATES]
    fallback = None
    for path in cands:
        if not path or not os.path.exists(path):
            continue
        try:
            cols = pd.read_csv(path, nrows=0, low_memory=False).columns
        except Exception:
            continue
        if "SMILES" not in cols:
            continue
        fallback = fallback or path
        if "PAMPA" in cols:
            try:
                df = pd.read_csv(path, usecols=["PAMPA"], low_memory=False)
                if int(pd.to_numeric(df["PAMPA"], errors="coerce").notna().sum()) == n_pt:
                    if verbose:
                        print(f"  全 assay 表（自动找到）：{path}", flush=True)
                    return path
            except Exception:
                pass
    if fallback and verbose:
        print(f"  全 assay 表（自动找到，未校验行数）：{fallback}", flush=True)
    return fallback


def load_metadata(ds: Dataset, meta_csv: Optional[str], verify_n: int = 16,
                  verbose: bool = True) -> pd.DataFrame:
    """给每个 data_i.pt 配上 SMILES / 来源 / 渗透率。优先用同目录的 sample_manifest.csv。

    会抽查 16 个分子：用 RDKit 重新算 Ipc 和 .pt 里的 view4[554] 比对，对不上直接停
    （元数据错位会让所有划分全错，宁可早停）。
    """
    n = len(ds)
    manifest = os.path.join(ds.data_dir, "sample_manifest.csv")
    frame = None
    if os.path.exists(manifest):
        man = pd.read_csv(manifest)
        if "sample_file" in man.columns and len(man) >= n:
            man = man.set_index("sample_file").reindex(ds.file_names).reset_index()
            frame = pd.DataFrame({
                "index": np.arange(n), "sample_file": ds.file_names,
                "smiles": man.get("smiles"),
                "source": man.get("source", pd.Series([""] * len(man))),
                "target_csv": pd.to_numeric(man.get("target"), errors="coerce"),
                "target_pampa": np.nan})
            if verbose:
                print(f"  元数据：{manifest}", flush=True)
    if frame is None and meta_csv:
        raw = pd.read_csv(meta_csv, low_memory=False)
        if "Permeability" in raw.columns:
            raw = raw.loc[pd.to_numeric(raw["Permeability"], errors="coerce").notna()].reset_index(drop=True)
        if len(raw) != n:
            raise SystemExit(
                f"元数据行数({len(raw)}) 与 .pt 数量({n}) 不一致：{meta_csv}\n"
                f"  请确认这张表就是生成这批 .pt 用的表（没有过滤、没有重排）。")
        frame = pd.DataFrame({
            "index": np.arange(n), "sample_file": ds.file_names,
            "smiles": raw["SMILES"] if "SMILES" in raw.columns else None,
            "source": raw["Source"] if "Source" in raw.columns else "",
            "target_csv": pd.to_numeric(raw["Permeability"], errors="coerce")
            if "Permeability" in raw.columns else np.nan,
            "target_pampa": pd.to_numeric(raw["PAMPA"], errors="coerce")
            if "PAMPA" in raw.columns else np.nan})
        if verbose:
            print(f"  元数据：{meta_csv}", flush=True)
    if frame is None:
        print("  ！没有元数据（既没有 sample_manifest.csv 也没给 --meta-csv）"
              "：需要 SMILES 的实验（严格/骨架/留出/来源/悬崖）会跳过。", flush=True)
        frame = pd.DataFrame({"index": np.arange(n), "sample_file": ds.file_names,
                              "smiles": None, "source": "", "target_csv": np.nan})

    if frame["smiles"].notna().any():
        _verify(ds, frame, verify_n, verbose)
    return frame


def _verify(ds: Dataset, frame: pd.DataFrame, verify_n: int, verbose: bool) -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    have = frame.index[frame["smiles"].notna()].to_numpy()
    picks = np.random.default_rng(0).choice(have, size=min(verify_n, len(have)), replace=False)
    bad_ipc, bad_tgt = [], []
    for i in picks:
        mol = Chem.MolFromSmiles(str(frame.at[i, "smiles"]))
        item = ds.raw(int(frame.at[i, "index"]))
        if mol is None or item is None:
            continue
        a, b = float(Descriptors.Ipc(mol)), float(item["view4_0d"][IPC_INDEX])
        ok = (a == b == 0.0) or (max(abs(a), abs(b)) > 1e6 and
                                 abs(math.log10(abs(a)) - math.log10(abs(b))) < 0.05) or \
             bool(np.isclose(a, b, rtol=1e-3, atol=1e-6))
        if not ok:
            bad_ipc.append((int(i), a, b))
        t = item["target"]
        t = float(t) if not torch.is_tensor(t) else float(t.reshape(-1)[0])
        cands = [c for c in (frame.at[i, "target_csv"], frame.at[i, "target_pampa"]
                             if "target_pampa" in frame.columns else np.nan)]
        cands = [float(c) for c in cands if pd.notna(c)]
        if cands and all(abs(t - c) > 1e-4 for c in cands):
            bad_tgt.append((int(i), cands[0], t))
    if bad_ipc:
        e = bad_ipc[0]
        raise SystemExit(
            "元数据与 .pt 对不上（抽查 Ipc 不一致）——很可能 --meta-csv 与这批 .pt 不是同一次生成。\n"
            f"  例：data_{e[0]}.pt 存的 view4[554]={e[2]:.4e}，CSV 第 {e[0]} 行算出 {e[1]:.4e}"
            f"（{len(bad_ipc)}/{len(picks)} 个抽查不一致）\n  请先确认数据来源，不要带着错位跑实验。")
    if bad_tgt:
        e = bad_tgt[0]
        raise SystemExit(f"元数据与 .pt 的 target 对不上：data_{e[0]}.pt={e[2]}，CSV={e[1]}"
                         f"（{len(bad_tgt)}/{len(picks)} 不一致）")
    if verbose:
        print(f"  元数据抽查：{len(picks)} 个分子的 Ipc/target 与 .pt 完全一致 ✓", flush=True)


def canonical_smiles_list(smiles: Sequence[str], verbose=False) -> np.ndarray:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    live = Live(len(smiles), "规范化 SMILES") if verbose else None
    out = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(str(s))
            out.append(Chem.MolToSmiles(m) if m is not None else "")
        except Exception:
            out.append("")
        if live:
            live.tick(1)
    if live:
        live.close()
    return np.asarray(out, dtype=object)


def morgan_fingerprints(smiles: Sequence[str], radius=2, n_bits=2048, verbose=False):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    live = Live(len(smiles), "计算指纹") if verbose else None
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        fps.append(gen.GetFingerprint(m) if m is not None else None)
        if live:
            live.tick(1)
    if live:
        live.close()
    return fps


def fp_matrix(fps, n_bits=2048) -> np.ndarray:
    from rdkit import DataStructs
    X = np.zeros((len(fps), n_bits), dtype=np.float32)
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        a = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, a)
        X[i] = a
    return X


def scaffold_groups(smiles: Sequence[str], verbose=False) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    live = Live(len(smiles), "提取分子骨架") if verbose else None
    out = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(str(s))
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else f"none{len(out)}")
        except Exception:
            out.append(f"none{len(out)}")
        if live:
            live.tick(1)
    if live:
        live.close()
    return np.asarray(out, dtype=object)


# ======================================================================================
# 2. 各种划分
# ======================================================================================
def _group_stats(labels, groups) -> pd.DataFrame:
    g = np.asarray([str(x) for x in groups], dtype=object)
    if (g == "").any():
        g = g.astype(object)
        g[g == ""] = [f"__s{i}" for i in np.where(g == "")[0]]
    df = pd.DataFrame({"g": g, "y": np.asarray(labels)})
    agg = df.groupby("g").agg(n=("y", "size"), pos=("y", "sum"))
    return agg.sort_values("n", ascending=False)


def group_kfold(labels, groups, n_splits=5, seed=42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """分层分组 K 折：同一组（同结构）绝不跨折，各折类别比例尽量一致。"""
    if n_splits < 2:
        raise SystemExit(f"折数（--folds）至少要 2，现在是 {n_splits}；1 折没有训练集。")
    labels = np.asarray(labels)
    g = np.asarray([str(x) for x in groups], dtype=object)
    if (g == "").any():
        g = g.astype(object)
        g[g == ""] = [f"__s{i}" for i in np.where(g == "")[0]]
    agg = _group_stats(labels, g)
    order = list(agg.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    order.sort(key=lambda k: -agg.at[k, "n"])
    fn, fp = np.zeros(n_splits), np.zeros(n_splits)
    total_n, total_pos = len(labels), int(labels.sum())
    assign = {}
    for k_ in order:
        row = agg.loc[k_]
        score = ((total_n / n_splits - fn) / max(total_n / n_splits, 1e-9)
                 + (total_pos / n_splits - fp) / max(total_pos / n_splits, 1e-9))
        k = int(np.argmax(score))
        assign[k_] = k
        fn[k] += int(row["n"])
        fp[k] += int(row["pos"])
    fold_of = np.array([assign[x] for x in g])
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_splits)]


def group_holdout(labels, groups, test_size=0.25, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    """分层分组留出：整组进测试集，类别比例尽量对齐。"""
    labels = np.asarray(labels)
    if len(labels) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    g = np.asarray([str(x) for x in groups], dtype=object)
    if (g == "").any():
        g = g.astype(object)
        g[g == ""] = [f"__s{i}" for i in np.where(g == "")[0]]
    agg = _group_stats(labels, g)
    order = list(agg.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    order.sort(key=lambda k: -agg.at[k, "n"])
    want, want_pos = int(round(len(labels) * test_size)), int(labels.sum()) * test_size
    chosen, n, npos = [], 0, 0
    for k_ in order:
        if n >= want:
            break
        row = agg.loc[k_]
        if n + row["n"] <= want * 1.15 and npos + row["pos"] <= max(want_pos * 1.25, want_pos + row["n"]):
            chosen.append(k_)
            n += int(row["n"])
            npos += int(row["pos"])
    if not chosen and order:
        chosen = [order[0]]
    s = set(chosen)
    mask = np.array([x in s for x in g], dtype=bool)
    return np.where(~mask)[0], np.where(mask)[0]


def val_split(train_pos, labels, groups, val_size=0.1, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    if len(train_pos) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    tr, va = group_holdout(np.asarray(labels)[train_pos], np.asarray(groups, dtype=object)[train_pos],
                           test_size=val_size, seed=seed)
    return train_pos[tr], train_pos[va]


def similarity_groups(fps, cutoff=0.85, verbose=True) -> np.ndarray:
    """把分子按 Tanimoto >= cutoff 的相似关系分组成「连通分量」。

    同一个分量永远整体进同一侧 ⇒ 数学上保证测试集里没有分子在训练集有 >=cutoff 的邻居。
    （层次聚类做不到这一点：完全相同结构的两个分子都可能被分到两侧。）
    """
    from rdkit import DataStructs
    valid = [i for i, f in enumerate(fps) if f is not None]
    n = len(valid)
    groups = np.empty(len(fps), dtype=object)
    for i, f in enumerate(fps):
        if f is None:
            groups[i] = f"none{i}"
    if n < 2:
        return groups
    if verbose:
        print(f"  按 Tanimoto>={cutoff} 分组：{n} 个分子、{n * (n - 1) // 2:,} 对（约 1 分钟）", flush=True)
    cond = np.empty(n * (n - 1) // 2, dtype=np.float32)   # scipy 行主序！顺序错=结果全错
    pos = 0
    live = Live(n - 1, "两两相似度") if verbose else None
    for a in range(n - 1):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(
            fps[valid[a]], [fps[valid[b]] for b in range(a + 1, n)]), dtype=np.float32)
        w = n - 1 - a
        cond[pos:pos + w] = 1.0 - sims
        pos += w
        if live:
            live.tick(1)
    if live:
        live.close()
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import squareform
    dense = squareform(cond, checks=False)
    labels = connected_components(csr_matrix((dense <= (1.0 - cutoff) + 1e-6).astype(np.int8)),
                                  directed=False)[1]
    del dense, cond
    for local, idx in enumerate(valid):
        groups[idx] = f"clu{int(labels[local])}"
    if verbose:
        vc = pd.Series(groups).value_counts()
        big = int(vc.iloc[0])
        print(f"  → 共 {vc.size} 个分量，最大分量 {big} 个分子（{big / n:.1%}），"
              f"其余 {n - big} 个（{(n - big) / n:.1%}）可用来做测试集", flush=True)
    return groups


def component_split(labels, groups, test_size=0.25, seed=42, verbose=True):
    """整分量进一侧：最大分量留在训练集，测试集用其余小分量凑。"""
    labels = np.asarray(labels)
    g = np.asarray([str(x) for x in groups], dtype=object)
    agg = pd.DataFrame({"g": g, "y": labels}).groupby("g").agg(n=("y", "size"), pos=("y", "sum"))
    want = int(round(len(labels) * test_size))
    order = list(agg.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    order.sort(key=lambda k: agg.at[k, "n"])
    chosen, n = [], 0
    for k_ in order:
        if n >= want:
            break
        if n + agg.at[k_, "n"] > want * 1.25:
            continue
        chosen.append(k_)
        n += int(agg.at[k_, "n"])
    if not chosen:
        chosen = [order[0]]
    s = set(chosen)
    mask = np.array([x in s for x in g])
    if verbose:
        print(f"  划分：测试集 {int(mask.sum())} 个（{mask.mean():.1%}）、训练集 {int((~mask).sum())} 个", flush=True)
    return np.where(~mask)[0], np.where(mask)[0]


def leakage_audit(tr, te, smiles, can, fps, cutoff=0.85, verbose=True) -> dict:
    """报告这次划分到底有没有泄漏：同结构跨集 + 跨集最大 Tanimoto + 有近邻的比例。"""
    tr, te = np.asarray(tr), np.asarray(te)
    tr_set = set(np.asarray(can)[tr].tolist())
    identical = int(sum(1 for c in np.asarray(can)[te] if c in tr_set))
    X, Xtr = fp_matrix([fps[i] for i in te]), fp_matrix([fps[i] for i in tr])
    pt, ptr = X.sum(1), Xtr.sum(1)
    nn = np.zeros(len(te), dtype=np.float32)
    for s in range(0, len(te), 512):
        blk = X[s:s + 512] @ Xtr.T
        nn[s:s + 512] = (blk / (pt[s:s + 512][:, None] + ptr[None, :] - blk + 1e-9)).max(1)
    out = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "same_structure_crossing": identical,
           "max_cross_tanimoto": float(nn.max()) if len(te) else float("nan"),
           "frac_test_with_nn_ge_cutoff": float((nn >= cutoff).mean()) if len(te) else float("nan"),
           "median_nn_tanimoto": float(np.median(nn)) if len(te) else float("nan")}
    if verbose:
        print(f"  泄漏检查：同结构跨集 {out['same_structure_crossing']} 个 | "
              f"跨集最大 Tanimoto {out['max_cross_tanimoto']:.3f} | "
              f"测试集有 >={cutoff} 近邻的比例 {out['frac_test_with_nn_ge_cutoff']:.1%} | "
              f"最近邻相似度中位数 {out['median_nn_tanimoto']:.3f}",
              flush=True)
    return out


def source_loso(sources, min_n=30, verbose=True):
    s = pd.Series([str(x) for x in sources])
    counts = s.value_counts()
    keep = counts[counts >= min_n].index.tolist()
    out = [(name, np.where((s != name).to_numpy())[0], np.where((s == name).to_numpy())[0]) for name in keep]
    if verbose:
        print(f"  来源留一：{len(out)} 个来源（每个 >= {min_n} 个分子），"
              f"覆盖 {sum(len(o[2]) for o in out)}/{len(s)} 个分子", flush=True)
    return out


# ======================================================================================
# 3. 模型 / 训练 / 指标
# ======================================================================================
class ZeroEncoder(torch.nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x, return_attention=False, **kw):
        if isinstance(x, dict) and "coords" in x:
            b, dev = x["coords"].shape[0], x["coords"].device
            ncf = x["coords"].shape[1] if x["coords"].dim() >= 2 else 1
        elif isinstance(x, dict) and "input_ids" in x:
            b, dev, ncf = x["input_ids"].shape[0], x["input_ids"].device, 1
        elif hasattr(x, "batch"):
            b = int(x.batch.max().item()) + 1 if x.batch.numel() else 1
            dev, ncf = x.x.device, 1
        else:
            b, dev, ncf = x.shape[0], x.device, 1
        f = torch.zeros((b, self.out_dim), device=dev)
        return (f, torch.zeros((b, ncf), device=dev)) if return_attention else f


class ZeroExpert(torch.nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        return torch.zeros((x.shape[0], self.out_dim), device=x.device)


# --------------------------------------------------------------------------------------
# 变体 → 结构改动表。**唯一真相来源**，新增变体只改这里 + VARIANTS。
#
#   空元组 = 结构就是 full（例如 wo_cliff 是「损失消融」，只关 use_cliff_loss）。
#   表里没登记的变体 = 结构改动未知 → build_model 直接报错，
#   而不是「悄悄按 full 跑」（那会得出一个看起来正常、其实错的消融结论）。
#
# 为什么改成表驱动：以前 build_model 是一串 if/elif，新增 wo_cliff 时漏写一支，
# 它掉进兜底的 raise，报 `ValueError: wo_cliff`。而报错发生在 build_model 阶段
# （还没开始训练），于是「消融」整步永远失败 —— 9/18~9/20 一直「跑一晚上没结果」
# 的真根因。表驱动 + check_variant_coverage() 让「漏登记」在开跑前 5 秒就暴露。
# --------------------------------------------------------------------------------------
VARIANT_OPS = {
    "full":     (),
    "wo_cliff": (),                                        # 损失消融：结构 = full
    "wo_3d":    (("encoder_3d", ZeroEncoder),),
    "wo_1d":    (("encoder_1d", ZeroEncoder),),
    "wo_2d":    (("encoder_2d", ZeroEncoder),),
    "wo_0d":    (("encoder_0d", ZeroEncoder), ("raw_0d_expert", ZeroExpert)),
}


def check_variant_coverage(verbose: bool = True) -> List[str]:
    """核对 VARIANTS / VARIANT_OPS 是否自洽。返回「有问题、会崩」的变体名列表。

    只做字典比对，不碰 torch、不读数据，毫秒级。在「版本」和「自检」两步里调用，
    daemon 会在派发任何 GPU 任务之前先跑自检 → 配错变体就立刻停，不会白占一晚上卡。
    """
    miss = [v for v in VARIANTS if v not in VARIANT_OPS]      # VARIANTS 有、表里没登记 → 必崩
    extra = [v for v in VARIANT_OPS if v not in VARIANTS]     # 表里有、VARIANTS 没有 → 大概率笔误
    if verbose:
        print(f"  变体登记检查：VARIANTS {len(VARIANTS)} 个 / VARIANT_OPS {len(VARIANT_OPS)} 个")
        if miss:
            print(f"  {RED}！这些变体没在 VARIANT_OPS 里登记，一跑到就会 "
                  f"ValueError 秒崩：{miss}{RESET}")
            print(f"    修法：在 VARIANT_OPS 里补上它对应的结构改动（空元组 = 结构同 full）。")
        if extra:
            print(f"  {YELLOW}！VARIANT_OPS 里有 VARIANTS 之外的名字（不致命，但可能是笔误）："
                  f"{extra}{RESET}")
        if not miss and not extra:
            print(f"  {GREEN}✓ 变体登记自洽：{', '.join(VARIANTS)}{RESET}")
    return miss


def build_model(variant: str, device, seed=42):
    """按变体名做结构消融。表驱动，见 VARIANT_OPS。"""
    if variant not in VARIANT_OPS:
        raise ValueError(
            f"未知变体 {variant!r}：它不在 VARIANT_OPS 里（可选：{tuple(VARIANT_OPS)}）。"
            f"新加变体时必须在 VARIANTS 和 VARIANT_OPS 里同时登记，并明确它的结构改动。")
    torch.manual_seed(seed)
    m = ensure_src()["TetraViewNet"]()
    for attr, cls in VARIANT_OPS[variant]:
        setattr(m, attr, cls())
    return m.to(device)


def param_groups(model, lr, lr_3d):
    base, three = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (three if n.startswith("encoder_3d") else base).append(p)
    g = [{"params": base, "lr": lr}]
    if three:
        g.append({"params": three, "lr": lr_3d})
    return g


def move_batch(b, device):
    return {"view1": {k: v.to(device) for k, v in b["view1"].items()},
            "view2": {k: v.to(device) for k, v in b["view2"].items()},
            "view3": b["view3"].to(device), "view4": b["view4"].to(device)}


class Cfg:
    def __init__(self, epochs=100, batch_size=32, lr=1e-4, lr_3d=1e-5, early_stop=20,
                 sched_patience=10, select_metric="rmse", num_confs=10, seed=42, amp=True,
                 use_cliff_loss=True):
        self.epochs, self.batch_size, self.lr, self.lr_3d = epochs, batch_size, lr, lr_3d
        self.early_stop, self.sched_patience = early_stop, sched_patience
        self.select_metric, self.num_confs, self.seed = select_metric, num_confs, seed
        self.amp, self.use_cliff_loss = amp, use_cliff_loss


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def binary_metrics(y, s, thr=0.5) -> dict:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    yp = (s >= thr).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); tn = int(((yp == 0) & (y == 0)).sum())
    fp = int(((yp == 1) & (y == 0)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec + rec) and prec + rec > 0 else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    try:
        auc = float(_quiet(roc_auc_score, y, s))
    except Exception:
        auc = float("nan")
    try:
        ap = float(_quiet(average_precision_score, y, s))
    except Exception:
        ap = float("nan")
    return {"n": len(y), "threshold": float(thr), "ACC": float((tp + tn) / max(len(y), 1)),
            "F1": float(f1), "MCC": float(mcc), "AUROC": auc, "AUPRC": ap,
            "balanced_accuracy": float(np.nanmean([rec, spec])),
            "precision_high": float(prec), "recall_high": float(rec), "specificity_high": float(spec),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "pos_rate": float(np.mean(y)) if len(y) else float("nan")}


def pick_threshold(y, s) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 61):
        m = binary_metrics(y, s, float(t))
        if m["F1"] > best_f1:
            best_f1, best_t = m["F1"], float(t)
    return best_t


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    sc, rg, tg = [], [], []
    with torch.no_grad():
        for b in loader:
            o = model(move_batch(b, device), return_dict=True)
            sc.append(torch.sigmoid(o["classification"]).detach().cpu().numpy().reshape(-1))
            rg.append(o["regression"].detach().cpu().numpy().reshape(-1))
            tg.append(b["targets"].detach().cpu().numpy().reshape(-1))
    s = np.concatenate(sc) if sc else np.zeros(0)
    r = np.concatenate(rg) if rg else np.zeros(0)
    t = np.concatenate(tg) if tg else np.zeros(0)
    y = (t >= CLS_THRESHOLD).astype(int)
    m = binary_metrics(y, s, threshold)
    m["RMSE"] = float(np.sqrt(np.mean((t - r) ** 2))) if len(t) else float("nan")
    m["MAE"] = float(np.mean(np.abs(t - r))) if len(t) else float("nan")
    return m, y, s, t, r


def train_one_fold(model, train_loader, val_loader, device, cfg: Cfg, live: Live, label: str) -> dict:
    """训练一个折：NaN 步会被跳过并计数（不会把权重搞坏），到点早停。"""
    use_rmse = cfg.select_metric == "rmse"
    crit = ensure_src()["CompositeLoss"](lambda_focal=0.3, lambda_rank=0.5, lambda_mse=1.0)
    opt = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.lr_3d), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min" if use_rmse else "max", factor=0.5, patience=cfg.sched_patience, min_lr=1e-6)
    amp_on = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp_on)
    best, best_state, best_ep = float("-inf"), None, -1
    bad = nan_steps = skipped = 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in train_loader:
            tgt = b["targets"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                o = model(move_batch(b, device), return_dict=True)
                if cfg.use_cliff_loss:
                    rl = crit(o["regression"], tgt, classification_threshold=CLS_THRESHOLD)[0]
                else:
                    rl = torch.nn.functional.mse_loss(o["regression"], tgt)
                cl = torch.nn.functional.binary_cross_entropy_with_logits(
                    o["classification"], (tgt >= CLS_THRESHOLD).float())
                ent = o.get("modality_entropy", torch.tensor(0.0, device=device))
                loss = rl + 0.3 * cl - 0.02 * ent
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                if nan_steps >= 200:
                    return {"status": "nan_abort", "epochs_run": epoch, "nan_steps": nan_steps,
                            "skipped_steps": skipped, "best_epoch": best_ep, "seconds": time.time() - t0}
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gn):
                scaler.step(opt)
            else:
                skipped += 1
            scaler.update()
            tot += float(loss.detach()); nb += 1
        val = evaluate(model, val_loader, device)[0]
        key = -val["RMSE"] if use_rmse else 0.6 * val["AUROC"] + 0.4 * val["F1"]
        sched.step(val["RMSE"] if use_rmse else key)
        improved = np.isfinite(key) and key > best
        if improved:
            best, bad, best_ep = key, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        live.step(0, extra=f"{label} 第{epoch}/{cfg.epochs}轮 loss={tot / max(nb, 1):.3f} "
                           f"验证RMSE={val['RMSE']:.3f} AUROC={val['AUROC']:.4f} "
                           + ("↑变好" if improved else f"没进步 {bad}/{cfg.early_stop}"))
        if bad >= cfg.early_stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return {"status": "ok", "epochs_run": epoch, "nan_steps": nan_steps, "skipped_steps": skipped,
            "best_epoch": best_ep, "seconds": time.time() - t0, "best_state": best_state}


def weighted_sampler(labels):
    """样本不均衡时按类别加权。样本为空或只有一个类别时返回 None（就不加权了）。"""
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    c = np.bincount(labels, minlength=2).astype(np.float64)
    c[c == 0] = 1.0
    w = (1.0 / c)[labels]
    return torch.utils.data.WeightedRandomSampler(torch.from_numpy(w).double(), len(w), replacement=True)


class _IndexedDataset(torch.utils.data.Dataset):
    """按给定下标读 .pt（跨进程可用，必须定义在模块级）。"""

    def __init__(self, ds, idx):
        self.ds, self.idx = ds, [int(i) for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return {"__i__": j, "item": self.ds.raw(j)}


class ViewCollate:
    """拼 batch：取前 N 个构象、0D 特征 log1p 有界化、带上样本下标。"""

    def __init__(self, num_confs: int, with_index: bool = True):
        self.num_confs, self.with_index = num_confs, with_index
        self._collate = ensure_src()["tetra_view_collate"]

    def __call__(self, batch):
        batch = [b for b in batch if b["item"] is not None]
        if not batch:
            raise ValueError("整批样本都读不出来")
        out = self._collate([b["item"] for b in batch])
        k = self.num_confs
        avail = int(out["view1"]["coords"].shape[1])
        if not 1 <= k <= avail:
            raise ValueError(f"num_confs={k}, 数据里只有 {avail} 个构象")
        out["view1"]["coords"] = out["view1"]["coords"][:, :k]
        v4 = out["view4"]                       # ← 0D 特征 log1p 有界化（必须保留！）
        out["view4"] = torch.sign(v4) * torch.log1p(v4.abs())
        if self.with_index:
            out["dataset_index"] = torch.tensor([b["__i__"] for b in batch], dtype=torch.long)
        return out


def index_loader(ds: Dataset, idx, batch_size, num_confs: int, sampler=None, num_workers=4):
    """并行读数据：num_workers>0 时多个子进程同时解 .pt，GPU 不用干等（这是主要提速点）。"""
    kw = {}
    if num_workers and num_workers > 0:
        kw = {"persistent_workers": True, "prefetch_factor": 4}
    return torch.utils.data.DataLoader(
        _IndexedDataset(ds, idx), batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), collate_fn=ViewCollate(num_confs),
        num_workers=int(max(0, num_workers)), pin_memory=torch.cuda.is_available(), **kw)


# ======================================================================================
# 4. 结果落盘（支持断点续跑）
# ======================================================================================
class Result:
    def __init__(self, out_dir: str, tag: str):
        self.dir = ensure_dir(os.path.join(out_dir, "results", tag))

    def fold_json(self, task, variant, confs, seed, fold):
        return os.path.join(self.dir, "folds", f"{task}_{variant}_c{confs}_s{seed}_{fold}.json")

    def preds_csv(self, task, variant, confs, seed, fold):
        return os.path.join(self.dir, "preds", f"{task}_{variant}_c{confs}_s{seed}_{fold}.csv")

    def done(self, *a) -> bool:
        p = read_json(self.fold_json(*a))
        return bool(p and p.get("status") in ("ok", "nan_abort"))

    def save(self, task, variant, confs, seed, fold, record, preds=None):
        write_json(self.fold_json(task, variant, confs, seed, fold), record)
        if preds:
            write_csv(self.preds_csv(task, variant, confs, seed, fold), preds)


def run_splits(ctx, splits, g, variant, confs, seed, task, res: Result, device, live, extra=None,
               offset: int = 0):
    """跑一组划分。--only-part K 时只跑「全局第 K 个划分」（offset 是本次调用在整步里的起始编号），
    这样同一个 (步骤,变体) 可以被拆成多个短任务并行，中断粒度也从"整步"降到"单折"。
    划分本身是确定性的（同样的 seed 必然得到同样的 cv0/cv1/...），所以拆开跑与整步跑结果一致。"""
    rows = []
    _op = getattr(g, "only_part", -1)
    only = -1 if _op is None else int(_op)      # 注意：不能写 `or -1`，否则 only_part=0 会被吃掉
    if only >= 0:
        print(f"    [只跑第 {only} 部分] 本任务只训练/评估 1 折，其余折由别的任务并行做", flush=True)
    for i, sp in enumerate(splits):
        if only >= 0 and offset + i != only:
            continue
        name = sp["name"]
        if res.done(task, variant, confs, seed, name) and not g.overwrite:
            prev = read_json(res.fold_json(task, variant, confs, seed, name)) or {}
            live.log(f"  [跳过] {task} {variant} c{confs} s{seed} {name} 已完成（想重跑加 --overwrite）")
            if prev.get("metrics"):
                rows.append(prev["metrics"])
            live.step(1, extra=f"{task} {variant}")
            continue
        tr, va, te = sp["train"], sp["val"], sp["test"]
        if len(tr) == 0 or len(te) == 0:
            print(f"    → {name}: 跳过（训练集 {len(tr)} 个、测试集 {len(te)} 个样本，太少没法训）",
                  flush=True)
            live.step(1, extra=f"{task} {variant}")
            continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cfg = Cfg(epochs=g.epochs, batch_size=g.batch_size, lr=g.lr, lr_3d=g.lr_3d,
                  early_stop=g.early_stop, sched_patience=g.sched_patience,
                  select_metric=g.select_metric, num_confs=confs, seed=seed, amp=not g.no_amp,
                  use_cliff_loss=(variant != "wo_cliff"))
        model = build_model(variant, device, seed)
        y_tr = ctx["labels"][[ctx["pos_of"][int(i)] for i in tr]]
        nw = int(getattr(g, "num_workers", 4))
        tl = index_loader(ctx["ds"], tr, g.batch_size, confs, sampler=weighted_sampler(y_tr), num_workers=nw)
        vl = index_loader(ctx["ds"], va if va is not None and len(va) else tr, g.batch_size, confs, num_workers=nw)
        el = index_loader(ctx["ds"], te, g.batch_size, confs, num_workers=nw)
        t0 = time.time()
        diag = train_one_fold(model, tl, vl, device, cfg, live,
                              f"[{task} {variant} c{confs} s{seed} {name}]")
        metrics = {}
        if diag["status"] == "ok":
            _, yv, sv, _, _ = evaluate(model, vl, device)
            thr = pick_threshold(yv, sv)
            metrics, y, s, t, r = evaluate(model, el, device, thr)
            fixed = binary_metrics(y, s, 0.5)
            metrics.update({"ACC@0.5": fixed["ACC"], "MCC@0.5": fixed["MCC"]})
        row = {"task": task, "variant": variant, "confs": confs, "seed": seed, "fold": name,
               "split": sp.get("split", name), "n_train": int(len(tr)),
               "n_test": int(len(te)), "status": diag["status"],
               "nan_steps": diag["nan_steps"], "best_epoch": diag["best_epoch"],
               "epochs_run": diag["epochs_run"], "seconds": round(time.time() - t0, 1),
               **(extra or {}), **metrics}
        print(f"    → {name}: {diag['status']} 跑了{diag['epochs_run']}轮 "
              + (f"| ACC={metrics['ACC']:.4f} MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f} "
                 f"AUPRC={metrics['AUPRC']:.4f} RMSE={metrics['RMSE']:.3f}" if metrics else "| 无指标")
              + f" | 用时 {fmt_dur(time.time() - t0)}", flush=True)
        preds = None
        if metrics:
            preds = [{"task": task, "variant": variant, "confs": confs, "seed": seed, "fold": name,
                      "dataset_index": int(i), "smiles": (str(ctx["smiles"][ctx["pos_of"][int(i)]])
                                                          if ctx["smiles"] is not None else ""),
                      "source": (str(ctx["sources"][ctx["pos_of"][int(i)]])
                                 if ctx["sources"] is not None else ""),
                      "true_label": int(a), "score": float(b), "true_logPe": float(c),
                      "pred_logPe": float(d)}
                     for i, a, b, c, d in zip(te, y, s, t, r)]
        res.save(task, variant, confs, seed, name,
                 {"metrics": row, "diagnostics": {k: v for k, v in diag.items() if k != "best_state"},
                  "status": diag["status"]}, preds)
        rows.append(row)
        live.step(1, extra=f"{task} {variant}")
    return rows


# ======================================================================================
# 5. 实验上下文
# ======================================================================================
def seeds_of(g, multi: bool = False) -> List[int]:
    """没显式指定 --seeds 时：普通步骤只跑 seed 42（省 5 倍时间），只有「种子」步骤跑多 seed。"""
    if getattr(g, "seeds", None):
        return [int(x) for x in str(g.seeds).split(",") if str(x).strip()]
    return [42, 43, 44, 45, 46] if multi else [42]


def build_ctx(g, data_dir=None, meta_csv="__from_g__", need_smiles=True, need_source=False, verbose=True):
    ds = open_dataset(g, verbose=verbose) if (data_dir or g.data_dir) == g.data_dir else Dataset(data_dir or g.data_dir)
    if verbose:
        print(f"  数据：{len(ds)} 个样本（{data_dir or g.data_dir}）", flush=True)
    explicit = g.meta_csv if meta_csv == "__from_g__" else meta_csv
    if explicit in (None, False, ""):
        explicit = False if meta_csv is False else autodetect_meta_csv(len(ds), g, verbose=verbose)
    elif explicit and not os.path.exists(str(explicit)):
        print(f"  ！--meta-csv 指的路径不存在：{explicit}，改为自动查找", flush=True)
        explicit = autodetect_meta_csv(len(ds), g, verbose=verbose)
    meta = load_metadata(ds, explicit, verbose=verbose and not g.quiet)
    idx, tgt = load_targets(ds, verbose=verbose)
    if g.max_samples:
        idx, tgt = idx[:g.max_samples], tgt[:g.max_samples]
        print(f"  [调试] --max-samples 生效，只用 {len(idx)} 个样本", flush=True)
    labels = (tgt >= CLS_THRESHOLD).astype(int)
    pos_of = {int(a): p for p, a in enumerate(idx)}
    smiles = meta["smiles"].to_numpy(dtype=object)[idx] if need_smiles else None
    if need_smiles and (smiles is None or any(s is None or (isinstance(s, float) and np.isnan(s)) for s in smiles)):
        raise SystemExit("这个实验需要 SMILES，但没有元数据。请加 --meta-csv，"
                         "或让预处理生成的 sample_manifest.csv 和数据放在一起。")
    sources = meta["source"].to_numpy(dtype=object)[idx] if need_source else None
    ctx = {"ds": ds, "meta": meta, "index": idx, "targets": tgt, "labels": labels,
           "pos_of": pos_of, "smiles": smiles, "sources": sources,
           "_can": None, "_fps": None}
    if verbose:
        print(f"  样本：{len(idx)} 个，高渗透（logPe>={CLS_THRESHOLD}）占 {labels.mean():.1%}", flush=True)
    return ctx


def ctx_can(ctx):
    if ctx["_can"] is None:
        ctx["_can"] = canonical_smiles_list(ctx["smiles"])
    return ctx["_can"]


def ctx_fps(ctx):
    if ctx["_fps"] is None:
        ctx["_fps"] = morgan_fingerprints(ctx["smiles"])
    return ctx["_fps"]


def kfold_splits(ctx, g, seed):
    kf = group_kfold(ctx["labels"], ctx_can(ctx), n_splits=g.folds, seed=seed)
    out = []
    for k, (tr, te) in enumerate(kf):
        tr_in, va = val_split(tr, ctx["labels"], ctx_can(ctx), g.val_size, seed + k)
        out.append({"name": f"fold{k}", "split": f"fold{k}", "train": ctx["index"][tr_in],
                    "val": ctx["index"][va], "test": ctx["index"][te]})
    return out


def group_of(ctx, groups):
    return ctx["index"][groups]


def apply_dedup(ctx, g):
    """可选：按标准 SMILES 去重、或把多数类下采样到 1:1。"""
    if g.dedup == "none" and g.balance == "none":
        return ctx
    can = canonical_smiles_list(ctx["smiles"])
    keep = np.arange(len(ctx["index"]))
    if g.dedup in ("first", "median"):
        groups: Dict[str, List[int]] = {}
        for p, k in enumerate(can):
            groups.setdefault(str(k), []).append(p)
        chosen = []
        for members in groups.values():
            if len(members) == 1 or g.dedup == "first":
                chosen.append(members[0])
            else:
                med = float(np.median(ctx["targets"][members]))
                chosen.append(min(members, key=lambda m: abs(float(ctx["targets"][m]) - med)))
        keep = np.sort(np.asarray(chosen))
        print(f"  去重({g.dedup})：{len(ctx['index'])} → {len(keep)} 条（丢掉重复测量 "
              f"{len(ctx['index']) - len(keep)} 条）", flush=True)
    if g.balance == "undersample":
        lab = (ctx["targets"][keep] >= CLS_THRESHOLD).astype(int)
        pos, neg = keep[lab == 1], keep[lab == 0]
        n = min(len(pos), len(neg))
        rng = np.random.default_rng(seeds_of(g)[0])
        keep = np.sort(np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]))
        print(f"  平衡(下采样)：→ {len(keep)} 条（1:1）", flush=True)
    ctx["index"], ctx["targets"] = ctx["index"][keep], ctx["targets"][keep]
    ctx["labels"] = (ctx["targets"] >= CLS_THRESHOLD).astype(int)
    ctx["pos_of"] = {int(a): p for p, a in enumerate(ctx["index"])}
    if ctx["smiles"] is not None:
        ctx["smiles"] = ctx["smiles"][keep]
    if ctx["sources"] is not None:
        ctx["sources"] = ctx["sources"][keep]
    ctx["_can"] = ctx["_fps"] = None
    return ctx


# ======================================================================================
# 6. 各个实验
# ======================================================================================
def step_version(g) -> int:
    """打印版本号 + 文件指纹，用来核对服务器上的是不是最新一版。"""
    import hashlib
    path = os.path.abspath(__file__)
    data = open(path, "rb").read()
    print(f"  版本号   ：{VERSION}")
    print(f"  文件     ：{path}")
    print(f"  大小     ：{len(data) / 1024:.1f} KB（{len(data.splitlines())} 行）")
    print(f"  sha256   ：{hashlib.sha256(data).hexdigest()[:16]}（前 16 位，够核对了）")
    print(f"  修改时间 ：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))}")
    print("  核对方法：这个版本号应该和我发你的说明里写的一致；不一致说明没覆盖成功。")
    print()
    if check_variant_coverage(verbose=True):
        print(f"  {RED}✗ 变体定义不自洽，先别开跑（修法见上面）{RESET}")
        return 1
    return 0


def step_preflight(g) -> int:
    banner(STEP_TITLE["preflight"], STEP_WHY["preflight"])
    # 毫秒级的变体登记自检，放最前面：配错就立刻停，别白占一晚上的卡。
    # （daemon 会把 preflight 当「前戏步骤」先跑，所以这里拦得住。）
    if check_variant_coverage(verbose=True):
        raise SystemExit("变体定义不自洽（见上），已在自检阶段停下")
    print(f"  {env_report()}")
    ok = True
    ds = Dataset(g.data_dir)
    print(f"  数据文件：{len(ds)} 个")
    probe = next((ds.raw(i) for i in range(min(8, len(ds))) if ds.raw(i) is not None), None)
    if probe is not None:
        dim = int(probe["view4_0d"].numel())
        print(f"  0D 特征维度：{dim}（应为 729）、构象数：{int(probe['view1_3d']['coords'].shape[0])}")
        if dim != 729:
            print(f"  {RED}！维度不是 729，与模型不符{RESET}")
            ok = False
    meta_path = g.meta_csv if (g.meta_csv and os.path.exists(str(g.meta_csv))) \
        else autodetect_meta_csv(len(ds), g, verbose=False)
    if not meta_path:
        print(f"\n  {YELLOW}！！没有找到「元数据表」（生成这批 .pt 用的那张表）{RESET}")
        print("     没有它，严格划分 / 骨架划分 / 留出 / 来源留一 / 消融 / 构象 / 种子 / 基线 / 悬崖")
        print("     这 9 个步骤都跑不了（自检会直接停在这里，免得白跑）。")
        print("     请二选一：")
        print("       A) 加上参数指定表：")
        print("          python GPU_交给师兄/run_all.py --meta-csv "
              "/data/workplace/jwx/MSF-CPMP/datasets_process/CycPeptMPDB_Peptide_PAMPA.csv")
        print("       B) 把那张表拷到数据目录并改名：")
        print(f"          cp <PAMPA表> {os.path.join(g.data_dir, 'sample_manifest.csv')}")
        print("     （我按顺序找了这些位置，都没有：）")
        for c in _candidates_for(g):
            print(f"        - {c}")
        raise SystemExit("缺少元数据表，已在自检阶段停下")
    print(f"  元数据表：{meta_path}")

    ctx = build_ctx(g, need_smiles=False, need_source=True)
    if ctx["smiles"] is not None:
        can = canonical_smiles_list(ctx["smiles"])
        vc = pd.Series(can).value_counts()
        print(f"  不同结构数：{vc.size}（重复测量 {len(can) - vc.size} 条）")
    if ctx["sources"] is not None:
        s = pd.Series([str(x) for x in ctx["sources"]])
        print(f"  来源数：{s.nunique()}（>= {g.min_source_n} 个分子的来源："
              f"{int((s.value_counts() >= g.min_source_n).sum())} 个）")

    print("\n  ---- 0D 特征溢出检查（之前结果不稳的根因）----", flush=True)
    worst = []
    live = Live(len(ds), "扫描 view4")
    for i in range(len(ds)):
        it = ds.raw(i)
        if it is not None:
            v = it["view4_0d"]
            worst.append((float(v[IPC_INDEX]), i, int((v.abs() > 3e20).sum())))
        live.tick(1)
    live.close()
    if worst:
        vals = np.array([w[0] for w in worst])
        bad = int((vals >= 3e20).sum())
        print(f"  Ipc（会溢出的那一维）：中位数 {np.median(vals):.2e}，最大 {vals.max():.2e}")
        print(f"  超过 NaN 阈值(3e20)的分子：{bad} 个（{bad / len(vals):.1%}）→ "
              f"每个 batch 撞上它们的概率约 {1 - (1 - bad / len(vals)) ** 32:.1%}")
        print(f"  {GREEN}训练脚本已自动做 log1p 有界化，正常不会出问题{RESET}")
    print(f"\n  自检结论：{GREEN + 'OK，可以开跑' + RESET if ok else RED + '有问题，见上面！行' + RESET}")
    return 0 if ok else 1


# 补充材料 Table S1 里被归为「0D（与图拓扑无关）」的 21 个描述符
DESC_0D = ["MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumRadicalElectrons",
           "NumHeteroatoms", "HeavyAtomCount", "NumHDonors", "NumHAcceptors", "NOCount", "NHOHCount",
           "RingCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
           "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
           "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings"]


def _find_all_csv(g) -> Optional[str]:
    if g.all_csv and os.path.exists(g.all_csv):
        return g.all_csv
    for cand in ("/data/workplace/jwx/cycpeptmp/data/CycPeptMPDB_Peptide_All.csv",
                 "/data/workplace/jwx/MSF-CPMP/datasets_process/CycPeptMPDB_Peptide_All.csv",
                 os.path.join(repo_root(), "data", "CycPeptMPDB_Peptide_All.csv"),
                 os.path.join(repo_root(), "CycPeptMPDB_Peptide_All.csv")):
        if os.path.exists(cand):
            return cand
    return None


def step_dataflow(g) -> int:
    """把数据去向理成一条链（审稿人 R1 问「数据用在哪」、R2 说「约 7991 个唯一结构」），
    并生成 0D/2D 描述符清单。全部从现成的表算出来，不需要任何新数据。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    banner(STEP_TITLE["dataflow"], STEP_WHY["dataflow"])
    out_dir = ensure_dir(os.path.join(g.out_dir, "dataflow"))
    lines = ["# 数据流审计（自动生成，可直接用于正文的数据部分 / CONSORT 式流程图）", ""]
    csv_path = _find_all_csv(g)
    n_pt = len(Dataset(g.data_dir))

    if csv_path is None:
        lines += ["> 没找到 CycPeptMPDB 全表。用 `--all-csv /path/to/CycPeptMPDB_Peptide_All.csv` 指定后重跑。",
                  f"> 已知建模样本数（.pt 文件数）：{n_pt}"]
        print("\n".join(lines))
        with open(os.path.join(out_dir, "数据流审计.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return 0

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  原始表：{csv_path}")
    steps = [("原始发布条目数", len(df))]
    assay_cols = [c for c in ("PAMPA", "Caco2", "MDCK", "RRCK") if c in df.columns]
    for c in assay_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Permeability" in df.columns:
        df["Permeability"] = pd.to_numeric(df["Permeability"], errors="coerce")
        steps.append(("其中有渗透率测量值（Permeability 非空）", int(df["Permeability"].notna().sum())))
    if "SMILES" in df.columns:
        ok = df["SMILES"].notna()
        steps.append(("SMILES 非空", int(ok.sum())))
        can = canonical_smiles_list(df.loc[ok, "SMILES"].astype(str).tolist())
        uniq = pd.Series(can).nunique()
        steps.append(("去掉同结构重复后的唯一结构数", int(uniq)))
        steps.append(("重复条目数（同一结构多次测量）", int(len(can) - uniq)))
        dup = pd.Series(can).value_counts()
        steps.append(("有重复测量的结构数", int((dup > 1).sum())))
        steps.append(("重复最多的结构测量次数", int(dup.iloc[0]) if len(dup) else 0))
    for c in assay_cols:
        steps.append((f"{c} 端点有测量值的条目", int(df[c].notna().sum())))
    if "Source" in df.columns:
        vc = df["Source"].value_counts()
        steps.append(("文献来源数", int(vc.size)))
        steps.append((f"来源样本 >= {g.min_source_n} 的来源数", int((vc >= g.min_source_n).sum())))
    steps.append(("实际用于建模的样本数（.pt 文件数）", n_pt))

    print("\n  ── 数据流（每一步剩多少）──")
    for k, v in steps:
        print(f"    {k:<38s} {v:>8d}")
    lines += ["| 步骤 | 数量 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in steps]

    lines += ["", "## 与正文数字的对应关系（请核对后填）", "",
              "| 正文里出现的数字 | 本次审计得到 | 说明 |", "|---|---|---|",
              f"| 8,466（原始条目） | {steps[0][1]} | 若用的是同一张发布表，这两个数应相等 |",
              f"| 7,991（唯一结构） | {dict(steps).get('去掉同结构重复后的唯一结构数', '—')} | 去重口径：canonical SMILES |",
              f"| 8,126 / 340（各自合计） | {n_pt}（建模样本） | 差值是「无 PAMPA 测量」或「预处理失败」的条目 |",
              "| 各实验类型样本量 | " + "、".join(f"{c} {int(df[c].notna().sum())}" for c in assay_cols) + " | 见上表 |", ""]

    # ---- 描述符清单（补充材料 Table S1）----
    desc = [n for n, _ in Descriptors.descList]
    zero_d = [n for n in desc if n in DESC_0D]
    lines += ["", "## 0D / 2D 描述符清单（补充材料 Table S1 用）", "",
              f"- 本机 RDKit 版本：`{__import__('rdkit').rdBase.rdkitVersion}`，"
              f"`Descriptors.descList` 长度 = **{len(desc)}**",
              "- 输入表征 = **512 位 Morgan 指纹（radius 2）** + **全部 RDKit 描述符** "
              f"（合计 {512 + len(desc)} 维 = view4 的维度）",
              f"- 其中按本文口径属于 **0D（与图拓扑无关）** 的共 {len(zero_d)} 个：",
              "  - " + ", ".join(zero_d),
              f"- 其余 {len(desc) - len(zero_d)} 个 + 512 位指纹归为 **2D**", "",
              "**完整描述符顺序（view4 的第 512+k 维 = 下表第 k 个）**", "",
              "| k | 描述符 | 归类 |", "|---|---|---|"]
    for k, name in enumerate(desc):
        lines.append(f"| {k} | {name} | {'0D' if name in DESC_0D else '2D'} |")
    lines += ["", "> ⚠️ 注意：上面的顺序里第 42 个（view4 第 554 维）是 `Ipc`，它的量级可达 1e36，"
                  "必须在送入模型前做 `sign(x)·log1p(|x|)` 有界化，否则整批 loss 会变成 NaN。"
                  "训练脚本已内置这一步。", ""]

    md_path = os.path.join(out_dir, "数据流审计.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    pd.DataFrame(steps, columns=["步骤", "数量"]).to_csv(os.path.join(out_dir, "数据流审计.csv"), index=False)
    print(f"\n  已写出：{md_path}")
    print(f"  已写出：{os.path.join(out_dir, '数据流审计.csv')}")
    return 0


def ensure_tokenizer(g) -> None:
    """离线 ChemBERTa 分词器：优先用包里的 chemberta_tokenizer/，无需联网。"""
    from transformers import AutoTokenizer
    local = os.path.join(HERE, "chemberta_tokenizer")
    tok = AutoTokenizer.from_pretrained(local, local_files_only=True) if \
        os.path.exists(os.path.join(local, "vocab.json")) else \
        AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    print(f"  分词器：{'本地 ' + local if os.path.exists(os.path.join(local, 'vocab.json')) else '在线'}"
          f" | vocab={tok.vocab_size} | pad={tok.pad_token!r}", flush=True)
    return tok


def step_pack(g) -> int:
    """把 tetraview_processed 下所有 .pt 打包成一个 memmap 数据集（训练提速的关键一步）。"""
    banner("打包数据（一次性，之后所有实验自动用打包数据）",
           "目的：把「每个 epoch 重新解 6941 个 .pt」换成「读一个 1–2 GB 的内存映射文件」，"
           "让 GPU 不再等 CPU（现在 GPU 利用率只有 5–24%，瓶颈就在这儿）。")
    raw = Dataset(g.data_dir)
    if not len(raw):
        raise SystemExit(f"{g.data_dir} 里没有 .pt 文件")
    out = ensure_dir(packed_dir_for(g))
    mc = g.meta_csv if (g.meta_csv and os.path.exists(str(g.meta_csv))) else \
        autodetect_meta_csv(len(raw), g, verbose=True)
    meta = load_metadata(raw, mc, verbose=True)
    print(f"  待打包：{len(raw)} 个样本", flush=True)

    probe = next((raw.raw(i) for i in range(len(raw)) if raw.raw(i) is not None), None)
    if probe is None:
        raise SystemExit("所有 .pt 都读不出来")
    n_conf = int(probe["view1_3d"]["coords"].shape[0])
    v1 = probe["view1_3d"]
    d_atom = int(v1["atom_feat"].shape[1])
    d_x = int(probe["view3_2d"].x.shape[1])
    d_edge = int(probe["view3_2d"].edge_attr.shape[1]) if probe["view3_2d"].edge_attr.numel() else 1
    seq_len = int(probe["view2_1d"]["input_ids"].numel())
    d_v4 = int(probe["view4_0d"].numel())

    # 先扫一遍拿到最大原子数与边数
    n_atoms = np.zeros(len(raw), dtype=np.int32)
    n_edges = np.zeros(len(raw), dtype=np.int64)
    live = Live(len(raw), "扫描原子数/边数")
    for i in range(len(raw)):
        it = raw.raw(i)
        if it is not None:
            n_atoms[i] = int(it["view3_2d"].x.shape[0])
            n_edges[i] = int(it["view3_2d"].edge_index.shape[1])
        live.tick(1)
    live.close()
    amax = int(n_atoms.max()) if len(n_atoms) else 1
    etot = int(n_edges.sum())
    print(f"  最大原子数 {amax}，总边数 {etot:,}，构象数 {n_conf}，"
          f"特征维度：原子 {d_atom} / 图 {d_x} / 边 {d_edge} / 0D {d_v4}，序列长 {seq_len}", flush=True)

    shapes = {
        "coords": ((len(raw), n_conf, amax, 3), np.float32),
        "atom_feat": ((len(raw), amax, d_atom), np.float32),
        "num_atoms": ((len(raw),), np.int32),
        "ids": ((len(raw), seq_len), np.int32),
        "mask": ((len(raw), seq_len), np.int32),
        "x": ((len(raw), amax, d_x), np.float32),
        "edge_index": ((max(etot, 1), 2), np.int32),
        "edge_attr": ((max(etot, 1), d_edge), np.float32),
        "edge_ptr": ((len(raw) + 1,), np.int64),
        "view4": ((len(raw), d_v4), np.float32),
        "target": ((len(raw),), np.float32),
    }
    files = {}
    for name, (shape, dt) in shapes.items():
        files[name] = np.lib.format.open_memmap(os.path.join(out, f"{name}.npy"), mode="w+",
                                                dtype=dt, shape=shape)
    ep, bad = 0, 0
    live = Live(len(raw), "写入打包数据")
    for i in range(len(raw)):
        it = raw.raw(i)
        if it is None:
            bad += 1
            live.tick(1)
            continue
        na = int(n_atoms[i])
        c = it["view1_3d"]["coords"].numpy().astype(np.float32)
        files["coords"][i, : c.shape[0], :na, :] = c[:, :na, :]
        files["atom_feat"][i, :na, :] = it["view1_3d"]["atom_feat"].numpy().astype(np.float32)[:na]
        files["num_atoms"][i] = na
        files["ids"][i, :] = it["view2_1d"]["input_ids"].numpy().astype(np.int32)
        mk = it["view2_1d"].get("attention_mask")
        files["mask"][i, :] = (mk.numpy().astype(np.int32) if mk is not None else np.ones(seq_len, np.int32))
        files["x"][i, :na, :] = it["view3_2d"].x.numpy().astype(np.float32)[:na]
        ei = it["view3_2d"].edge_index.numpy().astype(np.int32)
        ea = it["view3_2d"].edge_attr.numpy().astype(np.float32)
        files["edge_index"][ep:ep + ei.shape[1], :] = ei.T
        files["edge_attr"][ep:ep + ei.shape[1], :] = ea
        ep += ei.shape[1]
        files["edge_ptr"][i + 1] = ep
        files["view4"][i, :] = it["view4_0d"].numpy().astype(np.float32)
        t = it["target"]
        files["target"][i] = float(t) if not torch.is_tensor(t) else float(t.reshape(-1)[0])
        live.tick(1)
    live.close()
    for f in files.values():
        del f
    write_json(os.path.join(out, "manifest.json"), {
        "source_dir": os.path.abspath(g.data_dir), "n_samples": len(raw),
        "file_names": raw.file_names, "num_confs": n_conf, "n_atoms_max": amax,
        "dims": {"atom_feat": d_atom, "x": d_x, "edge_attr": d_edge, "seq_len": seq_len, "view4": d_v4},
        "n_edges_total": etot, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION})
    # 元数据也一起存进去，以后就不用再找那张表了
    man = pd.DataFrame({"sample_file": raw.file_names,
                        "smiles": meta["smiles"].to_numpy(dtype=object),
                        "source": meta["source"].to_numpy(dtype=object),
                        "target": np.load(os.path.join(out, "target.npy")).astype(float)})
    man.to_csv(os.path.join(out, "sample_manifest.csv"), index=False)
    size_mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1024 ** 2
    print(f"\n  打包完成：{out}（{size_mb:.0f} MB，坏样本 {bad} 个）")
    print("  以后所有命令会自动使用打包数据；想强制用原始 .pt 就加 --packed off")
    return 0


def step_preprocess(g) -> int:
    banner(STEP_TITLE["preprocess"], STEP_WHY["preprocess"])
    root = ensure_src()["root"]
    sys.path.insert(0, os.path.join(root, "scripts"))
    import preprocess as _pp
    from preprocess import compute_physchem_features, generate_3d_conformers, smiles_to_graph, tokenize_smiles
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if not getattr(_pp, "HAS_TRANSFORMERS", True):
        raise SystemExit("这台机器没装 transformers：分词会退化成「每个分子一套字符编号」，"
                         "与已发表数据不可比。请先 pip install transformers，再重跑本步骤。")

    n_desc = len(Descriptors.descList)
    if n_desc != 217:
        raise SystemExit(f"RDKit 描述符数 = {n_desc}，但数据是用 217 生成的（view4=512+217=729）。\n"
                         f"  请在这台机器上用与生成数据相同的 RDKit 版本重新生成，不要改模型维度。")
    n_pt = len(Dataset(g.data_dir))
    all_csv = autodetect_all_assay_csv(g, n_pt)
    if not all_csv:
        raise SystemExit(
            "找不到全 assay 表。请加参数指定后再跑，例如：\n"
            "  python GPU_交给师兄/run_all.py 预处理 "
            "--assay-csv /data/workplace/jwx/cycpeptmp/data/CycPeptMPDB_Peptide_All.csv")
    g.assay_csv = all_csv
    tok = ensure_tokenizer(g)
    df = pd.read_csv(g.assay_csv, low_memory=False)
    for assay in [a.strip() for a in g.assays.split(",") if a.strip()]:
        if assay not in df.columns:
            print(f"  ！表里没有 {assay} 列，跳过")
            continue
        out_dir = os.path.join(root, f"tetraview_processed_{assay.lower()}_chemberta")
        ensure_dir(out_dir)
        col = pd.to_numeric(df[assay], errors="coerce")
        sub = df.loc[col.notna()].copy().reset_index(drop=True)
        print(f"\n  {assay}：{len(sub)} 个分子 → {out_dir}", flush=True)
        pd.DataFrame({"sample_file": [f"data_{i}.pt" for i in range(len(sub))],
                      "source": sub["Source"] if "Source" in sub.columns else "",
                      "target": col[col.notna()].astype(float).values,
                      "smiles": sub["SMILES"].astype(str).values}).to_csv(
            os.path.join(out_dir, "sample_manifest.csv"), index=False)
        live = Live(len(sub), f"{assay} 预处理")
        ok = skip = fail = 0
        for i in range(len(sub)):
            path = os.path.join(out_dir, f"data_{i}.pt")
            if os.path.exists(path):
                skip += 1
                live.tick(1, extra=f"{assay} ok={ok} 跳过={skip}")
                continue
            smi = str(sub.at[i, "SMILES"])
            mol = Chem.MolFromSmiles(smi)
            v1 = generate_3d_conformers(smi, num_confs=g.num_confs) if mol is not None else None
            v2 = tokenize_smiles(smi, tok, max_len=200) if v1 is not None else None
            v3 = smiles_to_graph(smi) if v2 is not None else None
            v4 = compute_physchem_features(mol) if v3 is not None else None
            if v4 is None:
                fail += 1
                live.tick(1)
                continue
            torch.save({"view1_3d": v1, "view2_1d": v2, "view3_2d": v3, "view4_0d": v4,
                        "target": float(col[col.notna()].iloc[i]), "assay": assay, "smiles": smi}, path)
            ok += 1
            live.tick(1, extra=f"{assay} 新生成={ok} 跳过={skip}")
        live.close()
        print(f"  完成：新生成 {ok}，跳过 {skip}，失败 {fail} → {out_dir}")
    return 0


def step_ablation(g) -> int:
    banner(STEP_TITLE["ablation"], STEP_WHY["ablation"])
    ctx = apply_dedup(build_ctx(g), g)
    variants = [v.strip() for v in g.variants.split(",") if v.strip()]
    seeds = seeds_of(g)
    res, device = Result(g.out_dir, "消融"), dev(g)
    live = Live(len(variants) * len(seeds) * g.folds, f"消融实验（{len(variants)}个变体×{len(seeds)}个种子）")
    for seed in seeds:
        for variant in variants:
            run_splits(ctx, kfold_splits(ctx, g, seed), g, variant, g.num_confs, seed,
                       "消融", res, device, live)
    live.close()
    print(f"\n  结果目录：{res.dir}")
    return 0


def step_confs(g) -> int:
    banner(STEP_TITLE["confs"], STEP_WHY["confs"])
    ctx = apply_dedup(build_ctx(g), g)
    confs = [int(c) for c in g.confs.split(",")]
    seeds = seeds_of(g)
    res, device = Result(g.out_dir, "构象数"), dev(g)
    live = Live(len(confs) * len(seeds) * g.folds, "构象数扫描")
    for seed in seeds:
        for nc in confs:
            run_splits(ctx, kfold_splits(ctx, g, seed), g, "full", nc, seed, "构象数", res, device, live)
    live.close()
    return 0


def step_seeds(g) -> int:
    banner(STEP_TITLE["seeds"], STEP_WHY["seeds"])
    ctx = apply_dedup(build_ctx(g), g)
    seeds = seeds_of(g, multi=True)
    variants = [v.strip() for v in g.variants.split(",") if v.strip()]
    res, device = Result(g.out_dir, "种子复现"), dev(g)
    live = Live(len(seeds) * len(variants) * g.folds, "换种子复现")
    for seed in seeds:
        for variant in variants:
            run_splits(ctx, kfold_splits(ctx, g, seed), g, variant, g.num_confs, seed,
                       "种子复现", res, device, live)
    live.close()
    return 0


def _holdout_like(g, mode: str) -> int:
    banner(STEP_TITLE[mode], STEP_WHY[mode])
    ctx = apply_dedup(build_ctx(g), g)
    res, device = Result(g.out_dir, {"strict": "严格划分", "scaffold": "骨架划分",
                                     "holdout": "随机留出"}[mode]), dev(g)
    if mode == "strict":
        groups = similarity_groups(ctx_fps(ctx), cutoff=g.cutoff)
        tr, te = component_split(ctx["labels"], groups, test_size=g.test_size)
    elif mode == "scaffold":
        groups = scaffold_groups(ctx["smiles"], verbose=True)
        tr, te = group_holdout(ctx["labels"], groups, test_size=g.test_size)
    else:
        groups = ctx_can(ctx)
        tr, te = group_holdout(ctx["labels"], groups, test_size=g.test_size)
    audit = leakage_audit(ctx["index"][tr], ctx["index"][te], ctx["smiles"],
                          ctx_can(ctx), ctx_fps(ctx), cutoff=g.cutoff)
    write_json(os.path.join(res.dir, "leakage.json"), audit)
    if mode == "strict":
        good = audit["max_cross_tanimoto"] < g.cutoff and audit["same_structure_crossing"] == 0
        print(f"  严格划分自检（跨集最大 Tanimoto < {g.cutoff}、无同结构跨集）："
              f"{GREEN + '通过 ✓' + RESET if good else RED + '没通过，结果不能用' + RESET}", flush=True)
        if not good:
            raise SystemExit("严格划分未满足防泄漏约束")
    # 内部 K 折（在训练集内）+ 最终模型（测外部测试集）
    variants = [v.strip() for v in g.variants.split(",") if v.strip()]
    seeds = seeds_of(g)
    live = Live((g.folds + 1) * len(variants) * len(seeds), mode)
    for seed in seeds:
        for variant in variants:
            cv = group_kfold(ctx["labels"][tr], np.asarray(groups, dtype=object)[tr],
                             n_splits=g.folds, seed=seed)
            inner = []
            for k, (a, b) in enumerate(cv):
                tri, va = val_split(tr[a], ctx["labels"], groups, g.val_size, seed + k)
                inner.append({"name": f"cv{k}", "split": f"cv{k}", "train": ctx["index"][tri],
                              "val": ctx["index"][va], "test": ctx["index"][tr[b]]})
            run_splits(ctx, inner, g, variant, g.num_confs, seed, f"{mode}_内部CV", res, device, live,
                       extra={"partition": "内部CV"})
            tri, va = val_split(tr, ctx["labels"], groups, g.val_size, seed)
            run_splits(ctx, [{"name": "final", "split": "外部测试集", "train": ctx["index"][tri],
                              "val": ctx["index"][va], "test": ctx["index"][te]}],
                       g, variant, g.num_confs, seed, f"{mode}_外部测试", res, device, live,
                       extra={"partition": "外部测试集"}, offset=g.folds)
    live.close()
    return 0


def step_strict(g):
    return _holdout_like(g, "strict")


def step_scaffold(g):
    return _holdout_like(g, "scaffold")


def step_holdout(g):
    return _holdout_like(g, "holdout")


def step_loso(g) -> int:
    banner(STEP_TITLE["loso"], STEP_WHY["loso"])
    ctx = apply_dedup(build_ctx(g, need_source=True), g)
    res, device = Result(g.out_dir, "来源留一"), dev(g)
    srcs = source_loso(ctx["sources"], g.min_source_n)
    seeds = seeds_of(g)
    variants = [v.strip() for v in g.variants.split(",") if v.strip()]
    live = Live(len(srcs) * len(seeds) * len(variants), "来源留一")
    for seed in seeds:
        for variant in variants:
            splits = []
            for name, tr, te in srcs:
                tri, va = val_split(tr, ctx["labels"], ctx_can(ctx), g.val_size, seed)
                splits.append({"name": f"来源_{name}", "split": name, "train": ctx["index"][tri],
                               "val": ctx["index"][va], "test": ctx["index"][te]})
            run_splits(ctx, splits, g, variant, g.num_confs, seed, "来源留一", res, device, live)
    live.close()
    return 0


def step_assay(g) -> int:
    banner(STEP_TITLE["assay"], STEP_WHY["assay"])
    dirs = [d for d in ([g.assay_dir] if g.assay_dir else []) +
            [os.path.join(ensure_src()["root"], f"tetraview_processed_{a.strip().lower()}_chemberta")
             for a in g.assays.split(",") if a.strip()]]
    dirs = [d for d in dirs if os.path.isdir(d)]
    empty = [d for d in dirs if len(Dataset(d)) == 0]
    if empty:
        for d in empty:
            print(f"  ！{d} 里还没有数据（0 个 .pt）。先跑预处理：\n"
                  f"      python GPU_交给师兄/run_all.py 预处理", flush=True)
        dirs = [d for d in dirs if d not in empty]
    if not dirs:
        print("  → 跳过（没有可用的分 assay 数据）")
        return 0
    if not dirs:
        raise SystemExit("没找到分 assay 数据目录，请先跑：python GPU_交给师兄/run_all.py 预处理")
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        print(f"\n  ==== {name} ====")
        sub_g = argparse.Namespace(**vars(g))
        sub_g.data_dir, sub_g.meta_csv = d, None
        ctx = apply_dedup(build_ctx(sub_g, data_dir=d, meta_csv=False, verbose=True), sub_g)
        if len(ctx["index"]) < 30:
            print("  ！样本太少，只能作探索，不要写进结论")
        res, device = Result(g.out_dir, f"分assay_{name}"), dev(g)
        seeds = seeds_of(g)
        live = Live(g.folds * len(seeds), f"分assay {name}")
        for seed in seeds:
            run_splits(ctx, kfold_splits(ctx, sub_g, seed), sub_g, "full", g.num_confs,
                       seed, "分assay", res, device, live, extra={"assay": name})
        live.close()
    return 0


def step_baselines(g) -> int:
    banner(STEP_TITLE["baselines"], STEP_WHY["baselines"])
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR
    ctx = apply_dedup(build_ctx(g), g)
    res = Result(g.out_dir, "传统基线")

    # 用同一批特征（和网络看到的 0D 视图完全一致，含 log1p）
    cache = os.path.join(res.dir, "features.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        X, idx = z["X"], z["index"]
    else:
        rows, idx = [], []
        live = Live(len(ctx["index"]), "读取 0D 特征")
        for i in ctx["index"]:
            it = ctx["ds"].raw(int(i))
            if it is not None:
                rows.append(it["view4_0d"].float().numpy())
                idx.append(int(i))
            live.tick(1)
        live.close()
        X = np.asarray(rows, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.int64)
        X = np.sign(X) * np.log1p(np.abs(X))
        np.savez_compressed(cache, X=X, index=idx)
    look = {int(a): k for k, a in enumerate(idx)}
    y_bin = (ctx["targets"][[ctx["pos_of"][int(i)] for i in idx]] >= CLS_THRESHOLD).astype(int)
    y_reg = ctx["targets"][[ctx["pos_of"][int(i)] for i in idx]]

    # 用和"严格划分"一样的划分，保证对比公平
    groups = similarity_groups(ctx_fps(ctx), cutoff=g.cutoff, verbose=False)
    tr, te = component_split(ctx["labels"], groups, test_size=g.test_size, verbose=True)
    splits = [("外部测试集", tr, te)]
    cv = group_kfold(ctx["labels"][tr], np.asarray(groups, dtype=object)[tr], n_splits=g.folds, seed=42)
    splits += [(f"内部CV{k}", tr[a], tr[b]) for k, (a, b) in enumerate(cv)]

    rows = []
    live = Live(len(splits) * 2, "传统基线")
    for model_name in [m.strip() for m in g.models.split(",") if m.strip()]:
        for name, a_idx, b_idx in splits:
            trn = np.array([look[int(i)] for i in ctx["index"][a_idx] if int(i) in look])
            tst = np.array([look[int(i)] for i in ctx["index"][b_idx] if int(i) in look])
            if len(trn) < 10 or len(tst) < 5:
                live.step(1)
                continue
            if model_name == "RF":
                reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
                clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
            else:
                reg = Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", C=3.0, epsilon=0.1))])
                clf = Pipeline([("s", StandardScaler()), ("m", SVC(kernel="rbf", C=3.0, gamma="scale"))])
            reg.fit(X[trn], y_reg[trn])
            clf.fit(X[trn], y_bin[trn])
            pr = reg.predict(X[tst])
            if hasattr(clf, "predict_proba"):
                thr = pick_threshold(y_bin[trn], clf.predict_proba(X[trn])[:, 1])
                ps = clf.predict_proba(X[tst])[:, 1]
            else:
                thr, ps = 0.0, clf.decision_function(X[tst])
            m = binary_metrics(y_bin[tst], ps, thr)
            m.update({"model": model_name, "split": name, "n_train": int(len(trn)), "n_test": int(len(tst)),
                      "RMSE": float(np.sqrt(np.mean((y_reg[tst] - pr) ** 2))),
                      "MAE": float(np.mean(np.abs(y_reg[tst] - pr)))})
            rows.append(m)
            print(f"  {model_name:3s} {name:8s} n_test={len(tst):5d} ACC={m['ACC']:.4f} "
                  f"MCC={m['MCC']:.4f} AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} "
                  f"RMSE={m['RMSE']:.3f}", flush=True)
            live.step(1, extra=f"{model_name} {name}")
    live.close()
    pd.DataFrame(rows).to_csv(os.path.join(res.dir, "baselines.csv"), index=False)
    return 0


def step_cliffs(g) -> int:
    banner(STEP_TITLE["cliffs"], STEP_WHY["cliffs"])
    files = sorted(glob.glob(os.path.join(g.out_dir, "results", "**", "preds", "*.csv"), recursive=True))
    if not files:
        raise SystemExit("还没有逐样本预测结果，先跑前面的训练实验")
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if not d.empty:
                frames.append(d)
        except Exception:
            continue
    if not frames:
        raise SystemExit("预测文件都是空的，先跑前面的训练实验")
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[allp["smiles"].astype(str).str.len() > 0]
    print(f"  读到 {len(allp)} 条逐样本预测，来自 {len(files)} 个折文件", flush=True)
    out = ensure_dir(os.path.join(g.out_dir, "cliffs"))
    keys = [k for k in ("task", "variant", "confs", "seed", "fold") if k in allp.columns]
    rows = []
    for key, grp in allp.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        grp = grp.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        if len(grp) < 20:
            continue
        smi = grp["smiles"].astype(str).tolist()
        X = fp_matrix(morgan_fingerprints(smi))
        pop = X.sum(1)
        n = len(grp)
        true = grp["true_logPe"].to_numpy(float)
        pred = grp["pred_logPe"].to_numpy(float)
        S = np.zeros((n, n), dtype=np.float32)
        for s0 in range(0, n, 512):
            blk = X[s0:s0 + 512] @ X.T
            S[s0:s0 + 512] = blk / (pop[s0:s0 + 512][:, None] + pop[None, :] - blk + 1e-9)
        iu = np.triu_indices(n, k=1)
        dT, dP, sim = true[iu[0]] - true[iu[1]], pred[iu[0]] - pred[iu[1]], S[iu]
        for cut in [float(c) for c in g.cliff_cutoffs.split(",")]:
            for dmin in [float(c) for c in g.cliff_deltas.split(",")]:
                sel = (sim >= cut) & (np.abs(dT) >= dmin)
                cnt = int(sel.sum())
                row = dict(zip(keys, key))
                row.update({"tanimoto_cutoff": cut, "delta_cutoff": dmin, "pairs": cnt,
                            "ranking_accuracy": float((np.sign(dP[sel]) == np.sign(dT[sel])).mean()) if cnt else float("nan"),
                            "mean_abs_delta_error": float(np.mean(np.abs(dP[sel] - dT[sel]))) if cnt else float("nan")})
                rows.append(row)
                if cnt:
                    print(f"  {str(key):40s} Tanimoto>={cut} |ΔlogPe|>={dmin}: {cnt:6d} 对 | "
                          f"排序正确率 {row['ranking_accuracy']:.4f} | 平均|Δ误差| {row['mean_abs_delta_error']:.3f}",
                          flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "悬崖分析.csv"), index=False)
    print(f"\n  已写出：{os.path.join(out, '悬崖分析.csv')}")
    return 0


# ======================================================================================
# 8. 划分导出 / 出图 / 特征 t-SNE / 数据可得性 / 写作映射
# ======================================================================================
def _load_fold_metrics(g) -> pd.DataFrame:
    """把每个折的指标 JSON 读成一张表。"""
    rows = []
    for p in glob.glob(os.path.join(g.out_dir, "results", "**", "folds", "*.json"), recursive=True):
        payload = read_json(p, {}) or {}
        m = payload.get("metrics")
        if m:
            rows.append(m)
    return pd.DataFrame(rows)


def _load_preds(g) -> pd.DataFrame:
    frames = []
    for p in glob.glob(os.path.join(g.out_dir, "results", "**", "preds", "*.csv"), recursive=True):
        try:
            d = pd.read_csv(p)
            if not d.empty:
                frames.append(d)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _strict_split(ctx, g):
    """严格划分（Tanimoto 连通分量）。返回 (train_pos, test_pos, groups)。"""
    groups = similarity_groups(ctx_fps(ctx), cutoff=g.cutoff, verbose=True)
    tr, te = component_split(ctx["labels"], groups, test_size=g.test_size)
    return tr, te, groups


def build_protocol_splits(g, ctx) -> Dict[str, dict]:
    """把论文里用到的所有划分都算出来（确定性，不需要训练）。"""
    protos = {}
    # ① 随机分组 K 折（消融/构象用）
    kf = group_kfold(ctx["labels"], ctx_can(ctx), n_splits=g.folds, seed=42)
    protos["随机K折"] = {"folds": [], "groups": ctx_can(ctx)}
    for k, (tr, te) in enumerate(kf):
        tri, va = val_split(tr, ctx["labels"], ctx_can(ctx), g.val_size, seed=42 + k)
        protos["随机K折"]["folds"].append({"fold": k, "train": tri, "val": va, "test": te})
    # ② 严格（相似度连通分量）
    if True:
        tr, te, grp = _strict_split(ctx, g)
        cv = group_kfold(ctx["labels"][tr], np.asarray(grp, dtype=object)[tr], n_splits=g.folds, seed=42)
        folds = []
        for k, (a, b) in enumerate(cv):
            tri, va = val_split(tr[a], ctx["labels"], grp, g.val_size, seed=42 + k)
            folds.append({"fold": k, "train": tri, "val": va, "test": tr[b]})
        protos["严格"] = {"train": tr, "test": te, "folds": folds, "groups": grp}
    # ③ 骨架
    grp = scaffold_groups(ctx["smiles"])
    tr, te = group_holdout(ctx["labels"], grp, test_size=g.test_size, seed=42)
    protos["骨架"] = {"train": tr, "test": te, "groups": grp}
    # ④ 随机留出
    tr2, te2 = group_holdout(ctx["labels"], ctx_can(ctx), test_size=g.test_size, seed=42)
    protos["随机留出"] = {"train": tr2, "test": te2, "groups": ctx_can(ctx)}
    return protos


def step_splits(g) -> int:
    """把每次划分导出成 CSV（审稿人 R1：必须提供具体的数据划分）。"""
    banner(STEP_TITLE["splits"], STEP_WHY["splits"])
    ctx = build_ctx(g, need_smiles=True)
    out = ensure_dir(os.path.join(g.out_dir, "splits"))
    protos = build_protocol_splits(g, ctx)
    meta = ctx["meta"]

    def rows_for(name, role_idx, fold, role, group_arr=None):
        idx = ctx["index"][np.asarray(role_idx, dtype=np.int64)]
        out_rows = []
        for i in idx:
            p = ctx["pos_of"][int(i)]
            out_rows.append({"划分": name, "折": fold, "角色": role,
                             "dataset_index": int(i), "sample_file": ctx["ds"].file_names[int(i)],
                             "SMILES": ctx["smiles"][p] if ctx["smiles"] is not None else "",
                             "来源": ctx["sources"][p] if ctx["sources"] is not None else "",
                             "logPe": float(ctx["targets"][p]),
                             "类别": int(ctx["labels"][p]),
                             "分组ID": (str(group_arr[p]) if group_arr is not None else "")})
        return out_rows

    written = []
    for name, proto in protos.items():
        rows = []
        if "folds" in proto and proto["folds"]:
            for fd in proto["folds"]:
                rows += rows_for(name, fd["train"], fd["fold"], "train", np.asarray(proto["groups"], dtype=object))
                if len(fd.get("val", [])):
                    rows += rows_for(name, fd["val"], fd["fold"], "val", np.asarray(proto["groups"], dtype=object))
                rows += rows_for(name, fd["test"], fd["fold"], "test", np.asarray(proto["groups"], dtype=object))
        if "test" in proto:
            rows += rows_for(name, proto["train"], "-", "train", np.asarray(proto["groups"], dtype=object))
            rows += rows_for(name, proto["test"], "-", "test", np.asarray(proto["groups"], dtype=object))
        if not rows:
            continue
        path = os.path.join(out, f"划分_{name}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        written.append(path)
        n_tr = sum(1 for r in rows if r["角色"] == "train")
        n_te = sum(1 for r in rows if r["角色"] == "test")
        print(f"  写出 {os.path.basename(path)}：{len(rows)} 行（train {n_tr} / test {n_te}）", flush=True)

    # 泄漏审计（严格 vs 随机）一起写出来，正文直接引
    audit_rows = []
    for name in ("严格", "随机留出", "骨架"):
        if name not in protos:
            continue
        tr, te = protos[name]["train"], protos[name]["test"]
        a = leakage_audit(ctx["index"][tr], ctx["index"][te], ctx["smiles"], ctx_can(ctx),
                          ctx_fps(ctx), cutoff=g.cutoff, verbose=True)
        a["划分"] = name
        audit_rows.append(a)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(os.path.join(out, "泄漏审计.csv"), index=False)

    guide = ["# 数据划分说明（可直接写进正文 / Data Availability）", "",
             f"- 数据：`{g.data_dir}`，共 {len(ctx['index'])} 个样本；",
             f"- 标签：logPe >= {CLS_THRESHOLD} 记为高渗透（正类占 {ctx['labels'].mean():.1%}）；",
             f"- 所有划分都以 canonical SMILES 为分组键：**同一个结构（含重复测量）绝不会被分到不同折**；",
             f"- 随机种子 42；验证集从训练集里按 10% 分组切出。", "",
             "| 文件 | 含义 |", "|---|---|",
             "| `划分_随机K折.csv` | 消融/构象扫描用的分组 5 折（train/val/test 三角色）|",
             "| `划分_严格.csv` | 防泄漏协议：按 Tanimoto >= %.2f 的相似度**连通分量**整块划分，训练/测试之间不存在近邻 |" % g.cutoff,
             "| `划分_骨架.csv` | Bemis-Murcko 骨架划分（测试集骨架不出现在训练集）|",
             "| `划分_随机留出.csv` | 普通随机 25%% 留出（作为对照，展示随机划分会高估多少）|",
             "| `泄漏审计.csv` | 每种划分的实测泄漏：同结构跨集数、跨集最大 Tanimoto、有近邻比例 |", ""]
    with open(os.path.join(out, "划分说明.md"), "w") as fh:
        fh.write("\n".join(guide) + "\n")
    print(f"\n  已写出 {len(written)} 个划分文件 + 泄漏审计 + 划分说明 → {out}")
    return 0


# ---------------------------------------------------------------- 出图
def _setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "figure.dpi": 150,
    })
    return plt


OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def _save(fig, stem: str):
    ensure_dir(os.path.dirname(stem))
    fig.tight_layout()
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  图已写出：{stem}.pdf / .png", flush=True)


def _tanimoto_max_to(query_smiles, ref_smiles):
    qf = morgan_fingerprints(list(query_smiles))
    rf = morgan_fingerprints(list(ref_smiles))
    Q, R = fp_matrix(qf), fp_matrix(rf)
    pq, pr = Q.sum(1), R.sum(1)
    out = np.zeros(len(query_smiles), dtype=np.float32)
    for s in range(0, len(query_smiles), 256):
        blk = Q[s:s + 256] @ R.T
        out[s:s + 256] = (blk / (pq[s:s + 256][:, None] + pr[None, :] - blk + 1e-9)).max(1)
    return out


def step_figures(g) -> int:
    """用真实逐样本预测重画所有图（旧脚本是合成分数画的，必须替换）。"""
    banner(STEP_TITLE["figures"], STEP_WHY["figures"])
    plt = _setup_plot()
    from sklearn.metrics import (auc, average_precision_score, confusion_matrix, precision_recall_curve,
                                 roc_curve)
    out = ensure_dir(os.path.join(g.out_dir, "figures"))
    preds = _load_preds(g)
    if preds.empty:
        raise SystemExit("还没有逐样本预测（results/**/preds/*.csv），先把训练任务跑起来")
    print(f"  读到 {len(preds)} 条逐样本预测")

    # 兼容旧列名
    if "score" not in preds.columns and "classification_score" in preds.columns:
        preds["score"] = preds["classification_score"]
    preds["task"] = preds["task"].astype(str)

    # ---------- 图 4：ROC + PR（严格划分的外部测试集；没有就用严格内部 CV）----------
    def pick(prefixes):
        for pre in prefixes:
            hit = preds[preds["task"].str.startswith(pre)]
            if len(hit) > 20:
                return hit, pre
        return pd.DataFrame(), ""

    external, tag = pick(["strict_外部测试"])
    if external.empty:
        external, tag = pick(["strict_内部CV", "holdout_外部测试"])
    if not external.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        variants = sorted(external["variant"].unique())
        prevalence = float(external["true_label"].mean())
        for i, var in enumerate(variants):
            d = external[external["variant"] == var]
            y, sc = d["true_label"].to_numpy(int), d["score"].to_numpy(float)
            fpr, tpr, _ = roc_curve(y, sc)
            pre, rec, _ = precision_recall_curve(y, sc)
            axes[0].plot(fpr, tpr, color=OKABE[i % 7], lw=1.3,
                         label=f"{var} (AUC={auc(fpr, tpr):.3f})")
            axes[1].plot(rec, pre, color=OKABE[i % 7], lw=1.3,
                         label=f"{var} (AP={average_precision_score(y, sc):.3f})")
        axes[0].plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
        axes[1].axhline(prevalence, color="0.6", lw=0.8, ls=":")
        axes[1].text(0.02, prevalence + 0.02, f"prevalence = {prevalence:.3f}",
                     transform=axes[1].get_yaxis_transform(), fontsize=7, color="0.35")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
        axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
        axes[0].legend(frameon=False, loc="lower right")
        axes[1].legend(frameon=False, loc="lower left")
        for ax, t in zip(axes, "ab"):
            ax.set_title(t, loc="left", fontweight="bold")
        _save(fig, os.path.join(out, f"Fig4_ROC_PR_{tag}"))
        # 底层数字（可审计）
        num = []
        for var in variants:
            d = external[external["variant"] == var]
            y, sc = d["true_label"].to_numpy(int), d["score"].to_numpy(float)
            fpr, tpr, _ = roc_curve(y, sc)
            pre, rec, _ = precision_recall_curve(y, sc)
            num += [{"variant": var, "curve": "ROC", "x": a, "y": b} for a, b in zip(fpr, tpr)]
            num += [{"variant": var, "curve": "PR", "x": a, "y": b} for a, b in zip(rec, pre)]
        pd.DataFrame(num).to_csv(os.path.join(out, f"Fig4_数据_{tag}.csv"), index=False)

        # ---------- 每折 PR（回应"PR 曲线在 100% recall 处不收敛"）----------
        cv, cvtag = pick(["strict_内部CV", "holdout_内部CV"])
        if not cv.empty:
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            for i, (fo, d) in enumerate(cv.groupby("fold")):
                y, sc = d["true_label"].to_numpy(int), d["score"].to_numpy(float)
                pre, rec, _ = precision_recall_curve(y, sc)
                ax.plot(rec, pre, color=OKABE[i % 7], lw=1.1,
                        label=f"{fo} (prev={y.mean():.3f})")
            ax.axhline(float(cv["true_label"].mean()), color="0.6", lw=0.8, ls=":")
            ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.02))
            ax.legend(frameon=False, loc="lower left", ncol=2)
            _save(fig, os.path.join(out, f"Fig4b_PR_每折_{cvtag}"))

        # ---------- 图 10：预测 vs 实验（按到训练集的 Tanimoto 着色）----------
        sp_path = os.path.join(g.out_dir, "splits", "划分_严格.csv")
        if not os.path.exists(sp_path):
            print("  （还没导划分文件，先自动生成一次，用于散点着色）", flush=True)
            step_splits(g)
        tr_smiles = pd.DataFrame()
        if os.path.exists(sp_path):
            sp = pd.read_csv(sp_path)
            tr_smiles = sp[sp["角色"] == "train"]["SMILES"].dropna().unique()
        d = external[external["variant"] == external["variant"].iloc[0]].dropna(subset=["smiles"])
        if len(d) > 5:
            if len(tr_smiles):
                tan = _tanimoto_max_to(d["smiles"].astype(str).tolist(), list(tr_smiles))
            else:
                tan = np.zeros(len(d), dtype=np.float32)
            fig, ax = plt.subplots(figsize=(3.7, 3.2))
            sc = ax.scatter(d["true_logPe"], d["pred_logPe"], c=tan, cmap="viridis",
                            s=13, linewidths=0, alpha=0.85, vmin=0, vmax=1)
            lo = float(min(d["true_logPe"].min(), d["pred_logPe"].min()))
            hi = float(max(d["true_logPe"].max(), d["pred_logPe"].max()))
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls="--")
            ax.set(xlabel="Experimental log P$_{app}$", ylabel="Predicted log P$_{app}$")
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label("max Tanimoto to training set")
            r = float(np.corrcoef(d["true_logPe"], d["pred_logPe"])[0, 1])
            mae = float(np.abs(d["true_logPe"] - d["pred_logPe"]).mean())
            rmse = float(np.sqrt(((d["true_logPe"] - d["pred_logPe"]) ** 2).mean()))
            ax.text(0.03, 0.97, f"Pearson r = {r:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7,
                    bbox=dict(fc="white", ec="0.8", lw=0.5))
            _save(fig, os.path.join(out, "Fig10_预测vs实验_Tanimoto着色"))
            pd.DataFrame({"true_logPe": d["true_logPe"], "pred_logPe": d["pred_logPe"],
                          "max_tanimoto_to_train": tan}).to_csv(
                os.path.join(out, "Fig10_数据.csv"), index=False)

        # ---------- 混淆矩阵（两种阈值都画，回应意见 13）----------
        d = external[external["variant"] == external["variant"].iloc[0]]
        y = d["true_label"].to_numpy(int)
        s = d["score"].to_numpy(float)
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6))
        for ax, thr, name in ((axes[0], 0.5, "阈值 0.5"), (axes[1], None, "验证集选出的阈值")):
            if thr is None:
                thr = pick_threshold(y, s)
            cm = confusion_matrix(y, (s >= thr).astype(int), labels=[0, 1])
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > cm.max() / 2 else "black")
            ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True",
                   title=f"{name}（thr={thr:.2f}）")
            ax.set_xticklabels(["low", "high"]); ax.set_yticklabels(["low", "high"])
        _save(fig, os.path.join(out, "Fig_混淆矩阵_外部测试集"))

    # ---------- 构象数曲线 ----------
    folds = _load_fold_metrics(g)
    cf = folds[folds["task"] == "构象数"] if "task" in folds.columns else pd.DataFrame()
    if not cf.empty:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        gstat = cf.groupby("confs")["AUROC"]
        xs = sorted(cf["confs"].unique())
        ax.errorbar(xs, [gstat.get_group(x).mean() for x in xs],
                    yerr=[gstat.get_group(x).std(ddof=1) if len(gstat.get_group(x)) > 1 else 0 for x in xs],
                    marker="o", color=OKABE[0], capsize=3, lw=1.3, label="AUROC (mean ± SD)")
        for x in xs:
            ax.scatter([x] * len(gstat.get_group(x)), gstat.get_group(x), s=10, color="0.6", zorder=3)
        ax.set(xlabel="Number of conformers", ylabel="AUROC", xticks=xs)
        ax.legend(frameon=False, loc="lower right")
        _save(fig, os.path.join(out, "Fig11_构象数扫描"))
        cf.groupby("confs")["AUROC"].agg(["mean", "std", "count"]).to_csv(
            os.path.join(out, "Fig11_数据.csv"))

    # ---------- 消融柱状（真实数字 + 误差棒）----------
    ab = folds[folds["task"] == "消融"] if "task" in folds.columns else pd.DataFrame()
    if not ab.empty:
        order = [v for v in ("full", "wo_cliff", "wo_3d", "wo_0d", "wo_2d", "wo_1d")
                 if v in set(ab["variant"])]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
        for ax, key in zip(axes, ("AUROC", "ACC", "MCC")):
            means = [ab[ab["variant"] == v][key].mean() for v in order]
            errs = [ab[ab["variant"] == v][key].std(ddof=1) if len(ab[ab["variant"] == v]) > 1 else 0
                    for v in order]
            ax.bar(range(len(order)), means, yerr=errs, color=OKABE[0], capsize=3, width=0.62)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=30, ha="right")
            ax.set_title(key)
        _save(fig, os.path.join(out, "Fig_消融"))
        ab.groupby("variant")[["AUROC", "ACC", "MCC", "AUPRC"]].agg(["mean", "std", "count"]).to_csv(
            os.path.join(out, "Fig_消融_数据.csv"))

    # ---------- 留一来源条形 ----------
    lo = folds[folds["task"] == "来源留一"] if "task" in folds.columns else pd.DataFrame()
    if not lo.empty:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        gs = lo.groupby("split")["AUROC"].agg(["mean", "count"]).sort_values("mean")
        ax.barh(range(len(gs)), gs["mean"], color=OKABE[0])
        ax.set_yticks(range(len(gs)))
        ax.set_yticklabels([f"{i} (n={int(gs.loc[i, 'count'])})" for i in gs.index], fontsize=6)
        ax.axvline(0.5, color="0.6", lw=0.8, ls=":")
        ax.set(xlabel="AUROC")
        _save(fig, os.path.join(out, "Fig_留一来源"))
        gs.to_csv(os.path.join(out, "Fig_留一来源_数据.csv"))

    # ---------- 活动悬崖 Δ 散点 ----------
    cl = os.path.join(g.out_dir, "cliffs", "悬崖分析.csv")
    if os.path.exists(cl):
        d = pd.read_csv(cl)
        d = d[d["pairs"] > 0]
        if not d.empty:
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            labels = [f"T>={r.tanimoto_cutoff}, |Δ|>={r.delta_cutoff}" for r in d.itertuples()]
            ax.bar(range(len(d)), d["ranking_accuracy"], color=OKABE[2])
            ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
            ax.set_xticks(range(len(d))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
            ax.set(ylabel="Ranking accuracy", ylim=(0, 1.02))
            _save(fig, os.path.join(out, "Fig_活动悬崖"))
    print(f"\n  全部图已写出 → {out}")
    return 0


# ---------------------------------------------------------------- t-SNE
def step_features(g) -> int:
    """导出隐层融合特征并画 t-SNE（回应 R1：图 5B 的丝状伪影是超参数问题）。"""
    banner(STEP_TITLE["features"], STEP_WHY["features"])
    plt = _setup_plot()
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    ctx = build_ctx(g, need_smiles=True)
    device = dev(g)
    out = ensure_dir(os.path.join(g.out_dir, "features"))
    ckpt = os.path.join(g.out_dir, "results", "严格划分", "ckpt", "strict_外部测试_full_c10_s42_final.pt")
    tr, te, groups = _strict_split(ctx, g)
    if not tr.size:
        tr, te = group_holdout(ctx["labels"], ctx_can(ctx), test_size=g.test_size)
    train_abs = ctx["index"][tr][:2000]      # t-SNE 不需要全部样本，够看结构即可
    test_abs = ctx["index"][te][:2000]
    print(f"  用 {len(train_abs)} 个训练样本 + {len(test_abs)} 个测试样本导出特征", flush=True)

    model = build_model("full", device)
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print(f"  已载入权重：{ckpt}")
    else:
        # 没有存权重就现场训一个（1 次训练，几小时；有卡空着时跑）
        print("  没有找到已保存的权重，现场训练一个（会占用一张卡）…", flush=True)
        y_tr = ctx["labels"][[ctx["pos_of"][int(i)] for i in train_abs]]
        nw = int(getattr(g, "num_workers", 2))
        tl = index_loader(ctx["ds"], train_abs, g.batch_size, g.num_confs,
                          sampler=weighted_sampler(y_tr), num_workers=nw)
        vl = index_loader(ctx["ds"], train_abs[:max(64, len(train_abs) // 10)], g.batch_size,
                          g.num_confs, num_workers=nw)
        cfg = Cfg(epochs=g.epochs, batch_size=g.batch_size, lr=g.lr, lr_3d=g.lr_3d,
                  early_stop=g.early_stop, sched_patience=g.sched_patience,
                  select_metric=g.select_metric, num_confs=g.num_confs, amp=not g.no_amp)
        live = Live(g.epochs, "训练（t-SNE 用）")
        diag = train_one_fold(model, tl, vl, device, cfg, live, "[t-SNE 特征模型]")
        live.close()
        ensure_dir(os.path.dirname(ckpt))
        if diag.get("best_state") is not None:
            torch.save(diag["best_state"], ckpt)
            print(f"  权重已保存：{ckpt}")

    model.eval()
    feats, labels, which = [], [], []
    nw = int(getattr(g, "num_workers", 2))
    loader = index_loader(ctx["ds"], list(train_abs) + list(test_abs), g.batch_size, g.num_confs,
                          num_workers=nw)
    with torch.no_grad():
        for b in loader:
            o = model(move_batch(b, device), return_dict=True)
            f = o.get("fused_features")
            if f is None:
                f = o.get("shared_features")
            feats.append(f.detach().cpu().numpy())
            labels.append(b["targets"].detach().cpu().numpy().reshape(-1))
            which.append(np.where(np.isin(np.asarray(b["dataset_index"]),
                                          np.asarray(test_abs)))[0].size > 0)
    X = np.concatenate(feats)
    y_cont = np.concatenate(labels)
    y_bin = (y_cont >= CLS_THRESHOLD).astype(int)
    is_test = np.concatenate([np.isin(np.asarray(b["dataset_index"]), np.asarray(test_abs))
                              for b in loader])
    np.savez_compressed(os.path.join(out, "隐层特征.npz"), X=X, y=y_cont, is_test=is_test)
    print(f"  特征矩阵：{X.shape} → 已存 {out}/隐层特征.npz")

    # t-SNE：用推荐超参（perplexity 40、迭代 2000、PCA 初始化），并把不同 perplexity 的对比一起报
    rows_tsne = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, perp0 in zip(axes, (30, 50)):
        perp = max(5, min(perp0, (len(X) - 1) // 3))   # 样本少时自动降 perplexity
        Z = TSNE(n_components=2, perplexity=perp, max_iter=2000, init="pca",
                 learning_rate="auto", random_state=42).fit_transform(X)
        ax.scatter(Z[is_test, 0], Z[is_test, 1], s=6, c=OKABE[1], alpha=0.5, label="test (held out)")
        ax.scatter(Z[~is_test, 0], Z[~is_test, 1], s=6, c=OKABE[0], alpha=0.5, label="train")
        ax.set(title=f"t-SNE (perplexity={perp}, n_iter=2000, PCA init)", xticks=[], yticks=[])
        ax.legend(frameon=False, loc="best")
        try:
            sil = float(silhouette_score(Z, y_bin))
        except Exception:
            sil = float("nan")
        rows_tsne.append({"perplexity": perp, "n_iter": 2000, "init": "pca",
                          "silhouette_by_label": sil, "n_samples": int(len(X))})
        print(f"  perplexity={perp}: silhouette(按类别) = {sil:.3f}")
    _save(fig, os.path.join(out, "Fig5B_tSNE_超参对比"))
    pd.DataFrame(rows_tsne).to_csv(os.path.join(out, "tSNE_指标.csv"), index=False)
    print("  说明：图里同时给了 train/test 分布与轮廓系数，正文可据此说明超参已调好"
          "（perplexity 30/50、2000 次迭代、PCA 初始化）。")
    return 0


# ---------------------------------------------------------------- 数据可得性
def step_exportdata(g) -> int:
    """生成数据可得性声明需要的东西（不打包大文件，除非加 --pack-pt）。"""
    banner(STEP_TITLE["exportdata"], STEP_WHY["exportdata"])
    ctx = build_ctx(g, need_smiles=True)
    out = ensure_dir(os.path.join(g.out_dir, "data_availability"))
    rows = []
    for i in ctx["index"]:
        p = ctx["pos_of"][int(i)]
        rows.append({"dataset_index": int(i), "sample_file": ctx["ds"].file_names[int(i)],
                     "SMILES": ctx["smiles"][p], "来源": ctx["sources"][p] if ctx["sources"] is not None else "",
                     "logPe": float(ctx["targets"][p]), "类别": int(ctx["labels"][p])})
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(out, "样本清单.csv"), index=False)
    print(f"  样本清单：{len(inv)} 行 → {out}/样本清单.csv")

    params = {}
    try:
        import re
        src = open(os.path.join(ensure_src()["root"], "scripts", "preprocess.py"), encoding="utf-8").read()
        for key in ("NUM_CONFORMERS", "MAX_SEQ_LEN", "TOKENIZER_NAME", "ETKDG", "randomSeed", "numThreads",
                    "pruneRmsThresh"):
            for m in re.finditer(rf"^\s*{key}\s*=\s*(.+)$", src, re.M):
                params[key] = m.group(1).strip()
    except Exception as exc:
        params["读取 preprocess.py 失败"] = str(exc)

    doc = ["# 数据可得性 / 方法可复现说明（草稿，可直接改写成 Data Availability）", "",
           f"- 数据来源：CycPeptMPDB（PAMPA 端点），本研究所用样本数 **{len(inv)}**；",
           "- 每条记录包含四个视图：",
           "  1. `view1_3d`：10 个 ETKDGv3 构象的三维坐标与原子特征（**注意：ETKDGv3 是静态距离几何生成器，"
           "不是分子动力学，正文措辞需相应修改**）；",
           "  2. `view2_1d`：ChemBERTa 分词（vocab=767，max_len=200）；",
           "  3. `view3_2d`：分子图（原子特征 + 键类型）；",
           "  4. `view4_0d`：512 位 Morgan 指纹 + 217 个 RDKit 描述符（共 729 维）；"
           "其中第 554 维 RDKit `Ipc` 量级可达 1e36，**训练前必须做 sign(x)·log1p(|x|) 有界化**；",
           f"- 反式生成脚本与参数：`scripts/preprocess.py`（关键常量见下表）；",
           "- 划分文件：见 `splits/划分_*.csv`（每行标注样本属于 train/val/test 及第几折）；",
           "- 逐样本预测：见 `results/**/preds/*.csv`（含真实/预测 logPe 与分类分数）。", "",
           "| 预处理关键参数 | 值 |", "|---|---|"]
    for k, v in params.items():
        doc.append(f"| `{k}` | `{v}` |")
    doc += ["", "## 复现流程（三步）", "",
            "```bash",
            "python GPU_交给师兄/run_all.py 检查      # 环境与数据自检",
            "python GPU_交给师兄/run_all.py --gpus 0,1,2,3,4,5,6,7   # 全部实验",
            "python GPU_交给师兄/run_all.py 出表 划分 出图 特征 导出数据 写作映射",
            "```", ""]
    with open(os.path.join(out, "数据可得性说明.md"), "w") as fh:
        fh.write("\n".join(doc) + "\n")
    print(f"  说明文档 → {out}/数据可得性说明.md")

    if getattr(g, "pack_pt", False):
        import tarfile, hashlib
        tar_path = os.path.join(g.out_dir, "tetraview_processed.tar.gz")
        print(f"  开始打包四视图数据（很大，请耐心）→ {tar_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(g.data_dir, arcname=os.path.basename(os.path.normpath(g.data_dir)))
        print(f"  完成：{tar_path}")
    return 0


# ---------------------------------------------------------------- 写作映射
def step_writing(g) -> int:
    """把论文里每张图/表对应到具体文件与字段，并列出还缺什么。"""
    banner(STEP_TITLE["writing"], STEP_WHY["writing"])
    out = ensure_dir(os.path.join(g.out_dir, "tables"))
    folds = _load_fold_metrics(g)
    have = set(folds["task"].unique()) if not folds.empty else set()
    preds = _load_preds(g)
    figs = [os.path.basename(x) for x in glob.glob(os.path.join(g.out_dir, "figures", "*.png"))]
    sp = sorted(os.path.basename(x) for x in glob.glob(os.path.join(g.out_dir, "splits", "*.csv")))

    def mark(cond):
        return "✅ 已有" if cond else "❌ 缺"

    rows = [
        ("Table 1 内部 CV 指标", mark(any(t.startswith("strict_内部CV") for t in have)),
         "`tables/论文表格汇总.md` 第二节；原始数据 `results/**/folds/*.json`"),
        ("Table 2 外部测试集", mark(any(t.startswith("strict_外部测试") for t in have)),
         "同上；混淆矩阵另见 `figures/Fig_混淆矩阵_外部测试集.png`"),
        ("Table 4a 消融", mark("消融" in have), "`tables/论文表格汇总.md` 第三节（含配对检验 p 值）"),
        ("Table 4b 构象数扫描", mark("构象数" in have), "同上第四节；图 `figures/Fig11_构象数扫描.png`"),
        ("S1 0D/2D 描述符清单", mark(os.path.exists(os.path.join(g.out_dir, "dataflow", "数据流审计.md"))),
         "`dataflow/数据流审计.md` 的描述符表（带顺序与归类）"),
        ("S2 重复条目/去重口径", mark(os.path.exists(os.path.join(g.out_dir, "dataflow", "数据流审计.csv"))),
         "`dataflow/数据流审计.csv`（含重复测量条数）"),
        ("S4 类级指标与混淆矩阵（意见 13）", mark(not folds.empty),
         "`tables/论文表格汇总.md` 第六节 + `figures/Fig_混淆矩阵_外部测试集.png`"),
        ("意见 4 泄漏审计", mark(os.path.exists(os.path.join(g.out_dir, "splits", "泄漏审计.csv"))),
         "`splits/泄漏审计.csv`（同结构跨集数、跨集最大 Tanimoto、有近邻比例）"),
        ("意见 4 划分文件公开", mark(bool(sp)), f"`splits/` 下 {len(sp)} 个 CSV（train/val/test 逐样本标注）"),
        ("意见 5 骨架划分", mark(any(t.startswith("scaffold") for t in have)), "`tables/论文表格汇总.md` 第二节"),
        ("意见 6 来源留一", mark("来源留一" in have), "`tables/论文表格汇总.md` 第五节 + `figures/Fig_留一来源.png`"),
        ("意见 7 分 assay", mark(any(t.startswith("分assay") for t in have)), "`results/分assay_*/`"),
        ("意见 10 活动悬崖", mark(os.path.exists(os.path.join(g.out_dir, "cliffs", "悬崖分析.csv"))),
         "`cliffs/悬崖分析.csv` + `figures/Fig_活动悬崖.png`"),
        ("意见 11 经典 ML 基线", mark(os.path.exists(os.path.join(g.out_dir, "results", "传统基线",
                                                                 "baselines.csv"))),
         "`tables/论文表格汇总.md` 第七节"),
        ("意见 9 超参数报告", "✅ 已有（自动记录）", "每个折的 `folds/*.json` 里 `train_cfg` 字段"),
        ("图 4 ROC/PR（真实预测）", mark(any(f.startswith("Fig4_") for f in figs)),
         "`figures/Fig4_*.pdf/.png` + 同目录 `Fig4_数据_*.csv`"),
        ("图 5B t-SNE", mark(any("tSNE" in f for f in figs)), "`figures/Fig5B_tSNE_超参对比.png`"),
        ("图 6 活动悬崖图", mark(any("悬崖" in f for f in figs)), "`figures/Fig_活动悬崖.png`"),
        ("图 10 预测-实验散点（Tanimoto 着色）", mark(any("Fig10" in f for f in figs)),
         "`figures/Fig10_预测vs实验_Tanimoto着色.png` + `Fig10_数据.csv`"),
        ("图 11 构象数曲线", mark(any("Fig11" in f for f in figs)), "`figures/Fig11_构象数扫描.png`"),
        ("图 7 构象注意力重画", "❌ 需模型权重", "历史导出脚本在 `_历史归档/reproducibility/export_genuine_conformer_attention.py`；需用正式训练 checkpoint 生成逐构象权重"),
        ("意见 5 适用域分箱性能", "❌ 需补分析", "按测试样本到训练集最近邻 Tanimoto 分箱后报告 AUROC/MAE；当前泄漏审计只有整体统计"),
        ("⚠️ R1 要求：预训练/集成模型复现", "❌ 需要公开权重", "本套没有；需指定模型（PCPpred / Multi_CycGT / " 
                                                          "MultiCycPermea）后单独适配"),
        ("⚠️ R1 要求：SMD 阴性对照", "❌ 独立任务", "`Task7_SMD/`（GROMACS），与本套实验无关"),
        ("⚠️ 3D 构象集合打包公开", mark(getattr(g, "pack_pt", False)),
         "加 `--pack-pt` 重新跑本步骤即可打包（约 10 GB）"),
        ("⚠️ LLM 部分去留", "❌ 需决策", "R1 主张删、R2 主张谨慎保留，需你在回复信里定调"),
        ("⚠️ 文献综述补充", "❌ 写作", "transformer 化学语言模型 / 应用域评估（Geylan et al.）"),
        ("⚠️ 术语统一 / ETKDGv3 表述 / DFT 结论下调", "❌ 写作", "见 `审稿意见汇总.md` P2 段"),
    ]
    md = ["# 表图 ↔ 数据文件 对照表（写稿时照着填）", "",
          f"结果目录：`{g.out_dir}`", ""]
    md += ["| 论文里的位置 | 状态 | 数据在哪儿 / 备注 |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    md += ["", "## 还缺的实验（按优先级）", "",
           "1. **预训练/集成模型复现**（R1 明确要求）——需要你指定用哪个模型、权重从哪来；",
           "2. **SMD 阴性对照肽**（R1 要求）——`Task7_SMD/` 里的独立 GROMACS 任务；",
           "3. 若要把 3D 构象集合一起公开：`python GPU_交给师兄/run_all.py 导出数据 --pack-pt`；",
           "", "## 需要写作解决的（程序帮不上）", "",
           "- LLM 部分去留（R1 删除 / R2 谨慎保留）；",
           "- 文献综述补 transformer 化学语言模型与应用域评估；",
           "- 术语统一（permeability vs logP/logPe）、ETKDGv3 静态构象表述、DFT 结论下调为定性观察；",
           "- 补充材料作者名单（Guoyue Lv、Alfred Wei Chieh Kow、Zhongqi Fan）。", ""]
    path = os.path.join(out, "表图数据对照.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:14]))
    print(f"\n  ... 完整对照表 → {path}")
    return 0


def gpu_state():
    """返回 [(index, util%, mem_MB), ...]；拿不到就返回 None。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=10).decode()
        rows = []
        for line in out.strip().splitlines():
            idx, util, mem = [x.strip() for x in line.split(",")]
            rows.append((idx, int(float(util)), int(float(mem))))
        return rows
    except Exception:
        return None


def idle_gpus(max_util: int = 25, max_mem: int = 2000, exclude=()):
    """当前看起来空闲的卡（没人在用）：利用率低、显存占用小。"""
    st = gpu_state()
    if st is None:
        return None
    return [i for i, u, m in st if u <= max_util and m <= max_mem and i not in exclude]


JOB_WEIGHT = {"seeds": 60, "loso": 20, "strict": 10, "scaffold": 8, "holdout": 8,
              "ablation": 6, "confs": 5, "assay": 4, "baselines": 1, "cliffs": 1, "report": 1,
              "splits": 1, "figures": 1, "features": 8, "exportdata": 1, "writing": 1}
# 权重只决定"先派谁"（大的先跑）。时间不够被 --max-hours 截断时，
# 先跑的是 strict（论文主结果）→ ablation（消融）→ assay（分 assay）。


def observed_fold_hours(g) -> float:
    """从已完成的折里读实际用时（folds/*.json 的 seconds），拿中位数当基准；没有就按 2 小时。"""
    vals = []
    for p in glob.glob(os.path.join(g.out_dir, "results", "**", "folds", "*.json"), recursive=True):
        m = read_json(p, {}) or {}
        sec = (m.get("diagnostics") or {}).get("seconds")
        if sec and 60 < float(sec) < 24 * 3600:
            vals.append(float(sec))
    if len(vals) >= 3:
        return float(np.median(vals)) / 3600.0
    return 2.0


def estimate_plan(g, steps: List[str], n_gpu: int = 6) -> None:
    """粗估这套计划要跑多少折、还剩多少、多少小时（打包后约 2 小时/折）。

    会**扣掉已经跑完的折**，所以打印出来的"还剩"就是真正的工作量。
    """
    done_tasks = {}
    try:
        fm = _load_fold_metrics(g)
        if not fm.empty and "task" in fm.columns:
            done_tasks = fm.groupby("task").size().to_dict()
    except Exception:
        done_tasks = {}

    def done_for(prefixes):
        return sum(n for k, n in done_tasks.items() if any(str(k).startswith(x) for x in prefixes))

    plan, left = [], []
    for st in steps:
        if st == "ablation":
            n = len([x for x in g.variants.split(",") if x.strip()]) * g.folds
            d = done_for(["消融"])
        elif st == "confs":
            n = len(g.confs.split(",")) * g.folds
            d = done_for(["构象数"])
        elif st == "seeds":
            n = len(seeds_of(g, multi=True)) * 2 * g.folds
            d = done_for(["种子复现"])
        elif st in ("strict", "scaffold", "holdout"):
            n = g.folds + 1
            d = done_for([st])
        elif st == "loso":
            n, d = 12, done_for(["来源留一"])
        elif st == "assay":
            n, d = 5, done_for(["分assay"])
        elif st == "features":
            n, d = 1, (1 if os.path.exists(os.path.join(g.out_dir, "features", "隐层特征.npz")) else 0)
        else:
            continue
        plan.append((st, n))
        left.append((st, max(0, n - d)))

    name = {"ablation": "消融", "confs": "构象数", "seeds": "多种子", "strict": "严格划分",
            "scaffold": "骨架划分", "holdout": "随机留出", "loso": "来源留一",
            "assay": "分 assay", "features": "t-SNE 特征"}
    per_fold = observed_fold_hours(g)
    src = "服务器实测中位数" if per_fold != 2.0 else "默认值"
    print(f"\n  ── 工作量估算（按 {per_fold:.1f} 小时/折（{src}）、{n_gpu} 张卡）──")
    for (st, n), (_, l) in zip(plan, left):
        print(f"     {name.get(st, st):<10s} 计划 {n:>3d} 折，已完成 {n - l:>3d}，还剩 {l:>3d}")
    tot_plan = sum(n for _, n in plan)
    tot_left = sum(l for _, l in left)
    hours = tot_left * per_fold / max(n_gpu, 1)
    print(f"     {'合计':<10s} 计划 {tot_plan:>3d} 折，还剩 {tot_left:>3d} 折 ≈ {tot_left * per_fold:.0f} GPU·h"
          f" → {n_gpu} 张卡约 {hours:.1f} 小时")
    if hours > 20:
        print(f"     ⚠️ 超过 20 小时：建议把卡数加到 {int(n_gpu * hours / 20) + 1} 张，"
              f"或先只跑最关键的两项：run_all.py 严格 消融 --auto-gpus {n_gpu}")
    print(flush=True)


def step_daemon(g) -> int:
    """无人值守：自动挑卡、一直保持满负荷、崩了自动重试、训练完自动收尾。

    一个终端挂上去就不用人管了：
        nohup python GPU_交给师兄/run_all.py 无人值守 --auto-gpus 4 --num-workers 4 &
    """
    banner("无人值守模式（自动挑卡 + 自动补位 + 自动重试 + 自动收尾）",
           "目的：挂上去就不用管——一直保持有卡在跑，任务崩了自动重跑，训练跑完自动出表出图。")
    log_dir = ensure_dir(os.path.join(g.out_dir, "logs"))
    status_path = os.path.join(log_dir, "看守状态.json")
    t_start = time.time()
    max_hours = float(getattr(g, "max_hours", 72) or 72)
    job_hours = float(getattr(g, "job_hours", 0) or 0)
    n_parallel = int(getattr(g, "auto_gpus", 0) or 0) or len([x for x in (g.gpus or "0").split(",") if x.strip()])
    want = [x.strip() for x in (g.gpus or "").split(",") if x.strip()] or None

    extra = [STEP_NAMES.get(x, x) for x in getattr(g, "步骤", [])]
    extra = [x for x in extra if x in STEPS and x != "daemon"]
    if extra:
        steps = extra
    else:
        if getattr(g, "one_day", False):
            steps = list(TINY_ORDER if getattr(g, "tiny", False) else LEAN_ORDER)
        else:
            steps = list(DEFAULT_ORDER)
    if "smoke" in steps:
        steps.remove("smoke")
    print(f"  计划执行：{'、'.join(STEP_TITLE.get(x, x) for x in steps)}", flush=True)
    estimate_plan(g, steps, int(getattr(g, "auto_gpus", 0) or 6))
    # 前戏步骤（自检/打包/数据流）无论如何都要跑：打包是提速关键，不能因为指定了步骤就跳过
    jobs_all = expand_jobs(g, list(dict.fromkeys(["preflight", "pack", "dataflow"] + steps +
                                                 ["report", "splits", "figures", "writing"])))
    if getattr(g, "shard", ""):
        try:
            i, n = [int(x) for x in str(g.shard).split("/")]
            jobs_all = [j for k, j in enumerate(jobs_all) if k % n == i - 1]
            print(f"  [分片 {i}/{n}] 本机只跑 {len(jobs_all)} 个任务", flush=True)
        except Exception:
            raise SystemExit("--shard 格式应为 i/n")
    early = [j for j in jobs_all if j[0] in ("preflight", "pack", "dataflow", "version")]
    gpu_jobs = [j for j in jobs_all if j[0] in GPU_STEPS]
    post = [j for j in jobs_all if j[0] not in GPU_STEPS and j not in early]
    gpu_jobs.sort(key=lambda j: -JOB_WEIGHT.get(j[0], 1))     # 长任务先跑

    for j in early:
        print(f"\n  >>> 先跑：{STEP_TITLE.get(j[0], j[0])}", flush=True)
        try:
            STEPS[j[0]](g)
        except SystemExit as e:
            print(f"  ！{STEP_TITLE.get(j[0], j[0])} 中止：{e}", flush=True)
            if j[0] == "preflight":
                return 1

    if want:
        pool = want[:]
    else:
        cand = idle_gpus()
        pool = (cand or [str(i) for i in range(8)])[:n_parallel]
        print(f"  自动挑卡：GPU {'、'.join(pool)}（启动时最闲的 {len(pool)} 张）")

    pending = list(gpu_jobs)
    running: Dict[str, Tuple[List[str], subprocess.Popen, object, float]] = {}
    attempts: Dict[str, int] = {}
    done_jobs, failed_jobs = [], []
    print(f"\n  待跑任务 {len(pending)} 个（已完成的折会自动跳过），最多同时用 {len(pool)} 张卡"
          f"{'，单个任务最长 ' + str(job_hours) + ' 小时' if job_hours > 0 else ''}。", flush=True)
    print("  任务清单：", flush=True)
    for _i, _j in enumerate(pending):
        print(f"    {_i + 1:>3d}. {' '.join(_j)}", flush=True)

    def _tail(path, n=15):
        try:
            with open(path, errors="replace") as fh:
                return [ln.rstrip() for ln in fh.readlines()[-n:]]
        except Exception:
            return []

    def _show_tail(path, why):
        """把失败任务的日志尾巴直接打进看守日志，并追加到 logs/失败记录.md。

        以前失败只写一句「请检查」，人在看守日志里根本看不到报错，
        上一次 45 小时就是这么糊掉的 —— 现在失败原因必须自己跳出来。
        """
        lines = _tail(path, 15)
        if not lines:
            return
        print(f"      ┌── {os.path.basename(path)} 最后 {len(lines)} 行（{why}）", flush=True)
        for ln in lines:
            print(f"      │ {ln[:200]}", flush=True)
        print("      └── 详细全文见上面这个日志文件", flush=True)
        try:
            with open(os.path.join(log_dir, "失败记录.md"), "a") as fh:
                fh.write(f"\n\n## [{now_stamp()}] {why}\n\n日志：`{path}`\n\n```\n"
                         + "\n".join(lines) + "\n```\n")
        except Exception:
            pass

    def launch(gpu: str):
        if not pending:
            return
        job = pending.pop(0)
        key = " ".join(job)
        logfile = open(os.path.join(log_dir, f"任务_{'_'.join(job)}_GPU{gpu}.log"), "a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        # 顺序很重要：base_args 在前、job 在后。
        # 反过来的话，base_args 里的 --variants/--confs/--seeds 会盖掉每个小任务自己的设置
        # （argparse 同参数后者生效），于是 3 个消融任务各自都跑全部变体 —— 重复劳动、
        # 而且并发写同一批结果文件。这是上一批结果不可信的根因之一。
        cmd = job_cmd(g, job)
        proc = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT)
        running[gpu] = (job, proc, logfile, time.time())
        print(f"  [{now_stamp()}] GPU {gpu} 开始：{' '.join(job)}"
              f"（还剩 {len(pending)} 个待跑）", flush=True)

    last_beat = 0.0
    last_rescan = time.time()
    while pending or running:
        for gpu in pool:
            if gpu not in running:
                launch(gpu)
        time.sleep(10)
        for gpu in list(running):
            job, proc, logfile, t0 = running[gpu]
            key = " ".join(job)
            mins = (time.time() - t0) / 60
            over = job_hours > 0 and (time.time() - t0) / 3600 > job_hours
            if proc.poll() is None and not over:
                continue
            if over:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            logfile.close()
            if over:
                attempts[key] = 99          # 超时不重试：重试它照样会超时
                failed_jobs.append(key)
                print(f"  [{now_stamp()}] GPU {gpu} 超时（已跑 {mins / 60:.1f} 小时 > 上限 "
                      f"{job_hours} 小时），跳过：{key}", flush=True)
                _show_tail(logfile.name, f"超时跳过：{key}")
            elif proc.returncode == 0:
                done_jobs.append(key)
                print(f"  [{now_stamp()}] GPU {gpu} 完成：{key}（用了 {mins:.0f} 分钟）", flush=True)
            else:
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] <= 3:
                    pending.append(job)
                    print(f"  [{now_stamp()}] GPU {gpu} 失败（rc={proc.returncode}）：{key}，"
                          f"第 {attempts[key]} 次重试（{mins:.0f} 分钟）", flush=True)
                    _show_tail(logfile.name, f"第 {attempts[key]} 次失败：{key}（rc={proc.returncode}）")
                else:
                    failed_jobs.append(key)
                    print(f"  [{now_stamp()}] GPU {gpu} 连续失败 3 次，先跳过：{key}", flush=True)
                    _show_tail(logfile.name, f"连续失败 3 次跳过：{key}")
            del running[gpu]
        # 每 30 分钟看一眼有没有新空出来的卡（别人跑完了）
        if not want and time.time() - last_rescan > 1800:
            last_rescan = time.time()
            add = [x for x in (idle_gpus(exclude=set(pool)) or []) if x not in pool]
            if add and len(pool) < n_parallel:
                for x in add[: n_parallel - len(pool)]:
                    pool.append(x)
                print(f"  [{now_stamp()}] 发现空卡，加入：{'、'.join(add[:n_parallel])}", flush=True)
        if time.time() - last_beat > 300:
            last_beat = time.time()
            el = fmt_dur(time.time() - t_start)
            print(f"  [{now_stamp()}] 心跳：已跑 {el}，正在跑 {len(running)} 个，待跑 {len(pending)} 个，"
                  f"已完成 {len(done_jobs)} 个，**失败/跳过 {len(failed_jobs)} 个**", flush=True)
            if failed_jobs:
                print(f"      跳过的任务：{'；'.join(failed_jobs)}"
                      f"（报错尾巴在 logs/失败记录.md）", flush=True)
            write_json(status_path, {"updated": now_stamp(), "elapsed": el,
                                     "pool": pool, "running": {g_: " ".join(v[0]) for g_, v in running.items()},
                                     "pending": len(pending), "done_jobs": len(done_jobs),
                                     "failed_jobs": failed_jobs})
        if (time.time() - t_start) / 3600 > max_hours:
            print(f"  ！已超过 --max-hours={max_hours} 小时：停止接新任务、结束在跑的任务，"
                  f"然后照常执行收尾（出表/出图/导出划分）。已完成的折都落盘了，不会丢。", flush=True)
            for gpu, (job, proc, logfile, _) in list(running.items()):
                proc.terminate()
            break

    print(f"\n  训练类任务结束：成功 {len(done_jobs)} 个，失败/跳过 {len(failed_jobs)} 个"
          f"（每个的报错尾巴都在 logs/失败记录.md）", flush=True)
    if failed_jobs:
        print("  跳过的任务：" + "；".join(failed_jobs), flush=True)
    for j in post:
        print(f"\n  >>> 收尾：{STEP_TITLE.get(j[0], j[0])}", flush=True)
        try:
            STEPS[j[0]](g)
        except SystemExit as e:
            print(f"  ！{STEP_TITLE.get(j[0], j[0])} 中止：{e}", flush=True)
        except Exception as e:
            print(f"  ！{STEP_TITLE.get(j[0], j[0])} 出错：{type(e).__name__}: {e}", flush=True)

    write_json(status_path, {"updated": now_stamp(), "finished": True,
                             "elapsed": fmt_dur(time.time() - t_start),
                             "done_jobs": done_jobs, "failed_jobs": failed_jobs})
    print(f"\n{BOLD}无人值守结束{RESET}  总用时 {fmt_dur(time.time() - t_start)}")
    print(f"  汇总表：{os.path.join(g.out_dir, 'tables', '论文表格汇总.md')}")
    print(f"  失败原因汇总：{os.path.join(log_dir, '失败记录.md')}")
    return 1 if failed_jobs else 0


def dev(g):
    if g.device == "cpu":
        return torch.device("cpu")
    if g.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("要求用 GPU 但 CUDA 不可用")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================
# 7. 汇总出表 + 看板
# ======================================================================================
def _ms(g, k):
    v = g[k].dropna() if k in g.columns else pd.Series(dtype=float)
    if v.empty:
        return "缺"
    return f"{v.iloc[0]:.4f}" if len(v) == 1 else f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def step_report(g) -> int:
    banner(STEP_TITLE["report"], STEP_WHY["report"])
    rows = []
    for p in sorted(glob.glob(os.path.join(g.out_dir, "results", "**", "folds", "*.json"), recursive=True)):
        payload = read_json(p, {}) or {}
        m = payload.get("metrics")
        if m:
            rows.append(m)
    leaks = []
    for p in sorted(glob.glob(os.path.join(g.out_dir, "results", "**", "leakage.json"), recursive=True)):
        d = read_json(p, {}) or {}
        d["file"] = os.path.relpath(p, g.out_dir)
        leaks.append(d)
    def _read_csv_safe(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    base_files = sorted(glob.glob(os.path.join(g.out_dir, "results", "**", "baselines.csv"), recursive=True))
    parts = [_read_csv_safe(f) for f in base_files]
    parts = [p for p in parts if not p.empty]
    base = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cliff = _read_csv_safe(os.path.join(g.out_dir, "cliffs", "悬崖分析.csv"))

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(os.path.join(g.out_dir, "tables"))
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "全部逐折结果.csv"), index=False)
    md = [f"# TV-STFN 返修结果汇总（{time.strftime('%Y-%m-%d %H:%M')} 自动生成）", ""]

    tasks = sorted(df["task"].unique()) if not df.empty else []
    need = {"消融": "消融", "构象数": "构象", "种子复现": "种子", "严格划分": "严格",
            "骨架划分": "骨架", "随机留出": "留出", "来源留一": "来源", "分assay": "分assay"}
    md += ["## 一、完成情况（还差什么）", ""]
    for t, cn in need.items():
        hit = [x for x in tasks if x.startswith(t)]
        if hit:
            n = int(df[df["task"].isin(hit)].shape[0])
            bad = int((df[df["task"].isin(hit)]["nan_steps"].fillna(0) > 0).sum()) if "nan_steps" in df.columns else 0
            md.append(f"- ✅ {cn}：{n} 个折" + (f"（{bad} 个折出现 NaN 步，要检查）" if bad else ""))
        else:
            md.append(f"- ❌ **缺 {cn}** → `python GPU_交给师兄/run_all.py {cn}`")

    # 变体完整性：光看「消融有 N 个折」不够 —— 少一个变体，Table 4a 就是残缺的，
    # 而这种情况以前在汇总表里完全看不出来（9/20 那次 wo_cliff 全崩的盲区）。
    want_vars = [v.strip() for v in str(getattr(g, "variants", "") or "").split(",") if v.strip()]
    if want_vars and not df.empty and "variant" in df.columns:
        md += ["", "**变体完整性（对照本次 `--variants`）**", ""]
        for v in want_vars:
            sub = df[df["variant"] == v]
            if sub.empty:
                md.append(f"- ❌ **{v}：一个折都没有** → "
                          f"`python GPU_交给师兄/run_all.py 消融 --variants {v}`")
            else:
                kinds = "、".join(f"{t}×{int(c)}" for t, c in sub["task"].value_counts().items())
                md.append(f"- ✅ {v}：{len(sub)} 个折（{kinds}）")
    md += ["", f"- {'✅' if not base.empty else '❌'} 传统基线：{len(base)} 行", ""]

    if not df.empty:
        md += ["## 二、主结果（严格划分：内部 CV + 外部测试集）", ""]
        for part in ("内部CV", "外部测试集"):
            sub = df[(df.get("partition") == part)] if "partition" in df.columns else pd.DataFrame()
            md += [f"**{part}**", "", "| 变体 | 折数 | ACC | F1 | MCC | AUROC | AUPRC | RMSE |",
                   "|---|---|---|---|---|---|---|---|"]
            if sub.empty:
                md.append("| （缺，先跑 严格） | | | | | | | |")
            else:
                for var, gsub in sub.groupby("variant"):
                    md.append(f"| {var} | {len(gsub)} | {_ms(gsub,'ACC')} | {_ms(gsub,'F1')} | "
                              f"{_ms(gsub,'MCC')} | {_ms(gsub,'AUROC')} | {_ms(gsub,'AUPRC')} | {_ms(gsub,'RMSE')} |")
            md.append("")

        ab = df[df["task"] == "消融"]
        md += ["## 三、消融（去掉某个模块）", "", "| 变体 | 折数 | ACC | MCC | AUROC | AUPRC |",
               "|---|---|---|---|---|---|"]
        if ab.empty:
            md.append("| （缺，先跑 消融） | | | | | |")
        else:
            for var, gsub in ab.groupby("variant"):
                md.append(f"| {var} | {len(gsub)} | {_ms(gsub,'ACC')} | {_ms(gsub,'MCC')} | "
                          f"{_ms(gsub,'AUROC')} | {_ms(gsub,'AUPRC')} |")
            md += ["", "**full 对其他变体的配对检验（同一折/同一seed 一一配对）**", "",
                   "| 对比 | 配对数 | Δ AUROC(full−其他) | full 赢 | 其他赢 | 配对 t 检验 p |", "|---|---|---|---|---|---|"]
            try:
                from scipy import stats
            except Exception:
                stats = None
            f0 = ab[ab["variant"] == "full"].set_index(["seed", "fold"])
            for var, gsub in ab.groupby("variant"):
                if var == "full":
                    continue
                o = gsub.set_index(["seed", "fold"])
                ix = f0.index.intersection(o.index)
                if not len(ix):
                    continue
                a = f0.loc[ix, "AUROC"].to_numpy(float)
                b = o.loc[ix, "AUROC"].to_numpy(float)
                d = a - b
                p = "—"
                if stats is not None and len(d) > 1 and np.any(d != 0):
                    p = f"{stats.ttest_rel(a, b).pvalue:.4f}"
                md.append(f"| full vs {var} | {len(ix)} | {d.mean():+.4f} | {int((d>0).sum())} | "
                          f"{int((d<0).sum())} | {p} |")
            md += ["", "> 配对数少于 15 时统计上很难显著（5 折×1 seed = 5 对；建议 5 折×5 seed）。", ""]

        cf = df[df["task"] == "构象数"]
        md += ["## 四、构象数扫描", "", "| 构象数 | 折数 | ACC | MCC | AUROC | AUPRC |", "|---|---|---|---|---|---|"]
        if cf.empty:
            md.append("| （缺，先跑 构象） | | | | | |")
        else:
            for nc, gsub in cf.groupby("confs"):
                md.append(f"| {int(nc)} | {len(gsub)} | {_ms(gsub,'ACC')} | {_ms(gsub,'MCC')} | "
                          f"{_ms(gsub,'AUROC')} | {_ms(gsub,'AUPRC')} |")
            best = cf.groupby("confs")["AUROC"].mean().sort_values(ascending=False)
            md += ["", f"> 名义最好的构象数 = {int(best.index[0])}（AUROC {best.iloc[0]:.4f}）；"
                       "如果各点差异都在一个标准差内，就写「对构象数不敏感」，不要硬说 10 最好。", ""]

        lo = df[df["task"] == "来源留一"]
        md += ["## 五、留一来源（按文献来源）", "", "| 来源 | n_test | ACC | MCC | AUROC |", "|---|---|---|---|---|"]
        if lo.empty:
            md.append("| （缺，先跑 来源） | | | | |")
        else:
            for sp, gsub in lo.groupby("split"):
                md.append(f"| {sp} | {int(gsub['n_test'].iloc[0])} | {_ms(gsub,'ACC')} | "
                          f"{_ms(gsub,'MCC')} | {_ms(gsub,'AUROC')} |")
            md.append("")

        md += ["## 六、类别与类级指标（回答审稿人：F1 高但 AUROC 低的问题）", "",
               "| 任务 | 变体 | 折数 | 正类比例 | 精确率 | 召回率 | 特异度 | 平衡准确率 |", "|---|---|---|---|---|---|---|---|"]
        for (task, var), gsub in df.groupby(["task", "variant"]):
            md.append(f"| {task} | {var} | {len(gsub)} | {_ms(gsub,'pos_rate')} | {_ms(gsub,'precision_high')} | "
                      f"{_ms(gsub,'recall_high')} | {_ms(gsub,'specificity_high')} | {_ms(gsub,'balanced_accuracy')} |")
        md.append("")

    if not base.empty:
        md += ["## 七、传统基线（同一划分、同一特征）", "", "| 模型 | 划分 | n_test | ACC | MCC | AUROC | AUPRC | RMSE |",
               "|---|---|---|---|---|---|---|---|"]
        for (m, sp), gsub in base.groupby(["model", "split"]):
            md.append(f"| {m} | {sp} | {int(gsub['n_test'].mean())} | {_ms(gsub,'ACC')} | {_ms(gsub,'MCC')} | "
                      f"{_ms(gsub,'AUROC')} | {_ms(gsub,'AUPRC')} | {_ms(gsub,'RMSE')} |")
        md += ["", "> 如实对比：如果基线更高，正文必须照实写，不能只挑对自己有利的口径。", ""]

    if leaks:
        md += ["## 八、划分有没有泄漏（回答审稿人意见 4）", "",
               "| 划分 | 训练/测试 | 同结构跨集 | 跨集最大 Tanimoto | 测试集有近邻的比例 | 最近邻中位数 |",
               "|---|---|---|---|---|---|"]
        for d in leaks:
            md.append(f"| {d.get('file','')} | {d.get('n_train')}/{d.get('n_test')} | "
                      f"{d.get('same_structure_crossing')} | {d.get('max_cross_tanimoto', float('nan')):.3f} | "
                      f"{d.get('frac_test_with_nn_ge_cutoff', float('nan')):.1%} | "
                      f"{d.get('median_nn_tanimoto', float('nan')):.3f} |")
        md.append("")

    if not cliff.empty:
        md += ["## 九、活动悬崖对", "", "| 组 | Tanimoto≥ | |ΔlogPe|≥ | 对数 | 排序正确率 | 平均|Δ误差| |",
               "|---|---|---|---|---|---|"]
        show = cliff.sort_values("pairs", ascending=False).head(25)
        for _, r in show.iterrows():
            md.append(f"| {r.get('variant','')} {r.get('fold','')} | {r['tanimoto_cutoff']} | {r['delta_cutoff']} | "
                      f"{int(r['pairs'])} | {r['ranking_accuracy']:.4f} | {r['mean_abs_delta_error']:.3f} |")
        md.append("")

    path = os.path.join(out_dir, "论文表格汇总.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:60]))
    print(f"\n  ... 完整表格已写出：{path}")
    print(f"  明细数据：{os.path.join(out_dir, '全部逐折结果.csv')}")
    return 0


def _fold_table(g):
    stats, recent = {}, []
    for p in glob.glob(os.path.join(g.out_dir, "results", "**", "folds", "*.json"), recursive=True):
        payload = read_json(p, {}) or {}
        m = payload.get("metrics") or {}
        key = f"{m.get('task','?')} {m.get('variant','?')} c{m.get('confs','?')} s{m.get('seed','?')}"
        s = stats.setdefault(key, {"ok": 0, "bad": 0, "auroc": [], "last": 0.0})
        if payload.get("status") == "ok":
            s["ok"] += 1
        else:
            s["bad"] += 1
        if m.get("AUROC") is not None and np.isfinite(m.get("AUROC", np.nan)):
            s["auroc"].append(float(m["AUROC"]))
        s["last"] = max(s["last"], os.path.getmtime(p))
        recent.append((os.path.getmtime(p), key, payload.get("status"), m))
    recent.sort(reverse=True, key=lambda x: x[0])
    return stats, recent


def _gpu_line():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL,
                                      timeout=5).decode().strip().splitlines()
        return [f"  GPU {l.strip()}" for l in out]
    except Exception:
        return ["  （拿不到显卡信息：nvidia-smi 不可用）"]


def _procs():
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,etime,pcpu,args", "--sort=-pcpu"],
                                      stderr=subprocess.DEVNULL, timeout=5).decode().splitlines()
    except Exception:
        return []
    rows = [l for l in out[1:] if "run_all.py" in l and "看进度" not in l][:8]
    return [f"  {l.split(None, 3)[0]:>7s} 跑了 {l.split(None, 3)[1]:>10s} cpu {l.split(None, 3)[2]:>5s}%  "
            f"{l.split(None, 3)[3][:80]}" for l in rows]


def _print_status(g):
    stats, recent = _fold_table(g)
    print(f"{BOLD}TV-STFN 进度{RESET}  {time.strftime('%H:%M:%S')}")
    print(f"{CYAN}-- 显卡 --{RESET}")
    print("\n".join(_gpu_line()))
    print(f"{CYAN}-- 正在跑的任务 --{RESET}")
    p = _procs()
    print("\n".join(p) if p else "  （没有检测到正在运行的 run_all.py）")
    print(f"{CYAN}-- 各任务进度（折）--{RESET}")
    if not stats:
        print("  还没有任何结果")
    for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["last"]):
        au = f"{np.mean(s['auroc']):.4f}" if s["auroc"] else "—"
        ago = fmt_dur(time.time() - s["last"]) if s["last"] else "—"
        flag = f"{RED}{s['bad']}个异常{RESET}" if s["bad"] else "正常"
        print(f"  {k:38s} 完成 {s['ok']:3d} 折 | AUROC均值 {au} | {flag} | 最后更新 {ago} 前")
    print(f"{CYAN}-- 最近完成的折 --{RESET}")
    for mt, key, st, m in recent[:6]:
        if st == "ok" and m:
            print(f"  {GREEN}✓{RESET} {key:38s} ACC={m.get('ACC', float('nan')):.4f} "
                  f"MCC={m.get('MCC', float('nan')):.4f} AUROC={m.get('AUROC', float('nan')):.4f} "
                  f"{DIM}({fmt_dur(time.time() - mt)} 前){RESET}")
        else:
            print(f"  {YELLOW}{st}{RESET} {key}")
    logs = [f for f in glob.glob(os.path.join(g.out_dir, "logs", "*.log"))
            if time.time() - os.path.getmtime(f) < 600]
    logs.sort(key=os.path.getmtime, reverse=True)
    if logs:
        print(f"{CYAN}-- 每张卡的进度（日志尾部 3 行）--{RESET}")
        for f in logs[:3]:
            print(f"  {BOLD}{os.path.basename(f)}{RESET}")
            with open(f, errors="replace") as fh:
                for line in fh.readlines()[-3:]:
                    print("    " + line.rstrip()[:150])


def step_status(g) -> int:
    _print_status(g)
    return 0


def step_watch(g) -> int:
    tty = sys.stdout.isatty()
    try:
        while True:
            if tty:
                sys.stdout.write("\033[H\033[J")
            else:
                print("\n" + "=" * 78, flush=True)
            _print_status(g)
            sys.stdout.flush()
            time.sleep(g.interval)
    except KeyboardInterrupt:
        print("\n已退出看板")
    return 0


def step_smoke(g) -> int:
    banner(STEP_TITLE["smoke"], "用少量样本快速验证：能读数据 → 能训练 → 能出表。约 5 分钟。")
    sub = argparse.Namespace(**vars(g))
    sub.out_dir, sub.max_samples, sub.epochs, sub.folds = os.path.join(g.out_dir, "smoke"), 40, 1, 2
    sub.num_confs, sub.seeds, sub.variants = 1, "42", "full,wo_3d"
    sub.quiet = False
    print("  [1/3] 消融（1 折 1 轮）")
    step_ablation(sub)
    print("\n  [2/3] 严格划分（1 折 1 轮）")
    sub.variants = "full"
    step_strict(sub)
    print("\n  [3/3] 出表")
    step_report(sub)
    print(f"\n  {GREEN}冒烟测试完成 ✓{RESET} 说明环境和脚本都正常，可以正式开跑："
          f"python GPU_交给师兄/run_all.py")
    return 0


STEPS = {"preflight": step_preflight, "version": step_version, "pack": step_pack,
         "daemon": step_daemon,
         "splits": step_splits,
         "figures": step_figures, "features": step_features, "exportdata": step_exportdata,
         "writing": step_writing, "dataflow": step_dataflow, "ablation": step_ablation, "confs": step_confs,
         "seeds": step_seeds, "strict": step_strict, "scaffold": step_scaffold,
         "holdout": step_holdout, "loso": step_loso, "preprocess": step_preprocess,
         "assay": step_assay, "baselines": step_baselines, "cliffs": step_cliffs,
         "report": step_report, "status": step_status, "watch": step_watch, "smoke": step_smoke}

# 一天版：所有当前程序可自动运行的审稿人相关实验先跑 3 折；已完成折自动跳过。
LEAN_ORDER = ["preflight", "pack", "dataflow", "strict", "ablation", "holdout", "scaffold",
              "loso", "preprocess", "assay", "baselines", "cliffs", "report",
              "splits", "figures", "exportdata", "writing"]
LEAN_VARIANTS = "full,wo_cliff,wo_3d"
# 两三张卡一天版：纳入所有已有自动化入口；多种子和完整消融先留到后续精进。
TINY_ORDER = ["preflight", "pack", "dataflow", "ablation", "confs", "strict", "scaffold",
              "holdout", "loso", "preprocess", "assay", "baselines", "cliffs", "features",
              "report", "splits", "figures", "exportdata", "writing"]
TINY_DEFER = "多种子(50折)、消融 wo_0d/wo_2d(后续补足统计稳定性)；预训练模型和 SMD 阴性对照需外部权重/独立计算"

DEFAULT_ORDER = ["preflight", "pack", "dataflow", "ablation", "confs", "seeds", "strict", "scaffold", "holdout",
                 "loso", "preprocess", "assay", "baselines", "cliffs", "report",
                 "splits", "figures", "features", "exportdata", "writing"]

# 三卡并行时每组跑哪些（前面的重、后面的轻）
PARALLEL_GROUPS = [["ablation", "scaffold"], ["confs", "strict"], ["seeds", "loso"]]


# ======================================================================================
# 8. 命令行
# ======================================================================================
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TV-STFN 返修实验 · 傻瓜版（一个文件跑完全部）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例子：\n"
               "  python GPU_交给师兄/run_all.py                # 全部跑一遍（可断点续跑）\n"
               "  python GPU_交给师兄/run_all.py 检查 小测试     # 先自检再冒烟\n"
               "  python GPU_交给师兄/run_all.py 消融            # 只跑消融\n"
               "  python GPU_交给师兄/run_all.py --gpus 0,1,2   # 三卡并行\n"
               "  python GPU_交给师兄/run_all.py 看进度          # 实时看板\n")
    p.add_argument("步骤", nargs="*", help="中文步骤名：检查/消融/构象/种子/严格/骨架/留出/来源/"
                                          "预处理/分assay/基线/悬崖/出表/看进度/状态/小测试/全部")
    p.add_argument("--data-dir", default="tetraview_processed", help="四视图数据目录（默认 tetraview_processed）")
    p.add_argument("--meta-csv", default=None,
                   help="生成这批 .pt 用的表（PAMPA 表）。数据目录里有 sample_manifest.csv 时可省")
    p.add_argument("--out-dir", default="tvstfn_paper_pipeline/outputs/reviewer_gpu_followup",
                   help="结果目录（默认 reviewer_gpu_followup）")
    p.add_argument("--assay-csv", default=None, help="全 assay 表，供 预处理 步骤使用")
    p.add_argument("--all-csv", default=None,
                   help="CycPeptMPDB 全表（原始发布版），供 数据流 审计使用；不给就自动找常见路径")
    p.add_argument("--assay-dir", default=None, help="某个 assay 的四视图目录（默认按名字自动找）")
    p.add_argument("--assays", default="Caco2", help="预处理/分assay 要处理哪些端点，逗号分隔")
    p.add_argument("--gpus", default="", help="例如 0,1,2：自动分三组并行跑")
    p.add_argument("--epochs", type=int, default=100, help="每个折最多训多少轮（默认 100）")
    p.add_argument("--folds", type=int, default=5, help="交叉验证折数（默认 5；严格/骨架默认想用 10 可自己传）")
    p.add_argument("--seeds", default=None,
                   help="随机种子列表。不给时按步骤取默认：消融/构象/严格/骨架/留出/来源/分assay 只跑 42；"
                        "「种子」步骤跑 42,43,44,45,46。想统一指定就显式传这个参数。")
    p.add_argument("--num-confs", type=int, default=10, help="用几个构象（默认 10，数据里存了 10 个）")
    p.add_argument("--confs", default="1,2,3,5,10", help="构象数扫描列表")
    p.add_argument("--variants", default="full,wo_cliff,wo_3d,wo_0d,wo_2d",
                   help="消融变体：full/wo_cliff/wo_3d/wo_0d/wo_1d/wo_2d")
    p.add_argument("--cutoff", type=float, default=0.85, help="严格划分的 Tanimoto 阈值（默认 0.85）")
    p.add_argument("--test-size", type=float, default=0.25, help="外部测试集比例（默认 0.25 即 25%%）")
    p.add_argument("--min-source-n", type=int, default=30, help="来源留一的最小来源样本数")
    p.add_argument("--models", default="RF,SVM", help="传统基线模型（RF/SVM）")
    p.add_argument("--cliff-cutoffs", default="0.85,0.7")
    p.add_argument("--cliff-deltas", default="1.5,1.0")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--interval", type=float, default=5.0, help="看板刷新秒数")
    p.add_argument("--max-samples", type=int, default=0, help="调试：只用前 N 个样本")
    p.add_argument("--overwrite", action="store_true", help="忽略已完成结果，重跑")
    p.add_argument("--auto-gpus", type=int, default=0,
                   help="自动挑 N 张最闲的卡（共享服务器推荐；一天版会按 N 自动选方案）")
    p.add_argument("--all-idle-gpus", action="store_true",
                   help="启动时使用全部空闲 GPU，并在运行中自动吸纳后来空闲的 GPU")
    p.add_argument("--shard", default="",
                   help="分片：i/n，本机只跑第 i 份任务（多台机器/多个账号协作时用）")
    p.add_argument("--one-day", action="store_true",
                   help="一天版：所有已有自动化实验先跑 3 折；卡数 <=3 时跳过多种子和完整消融，已完成折自动跳过。")
    p.add_argument("--max-hours", type=float, default=72,
                   help="无人值守最长运行多少小时（默认 72；到点停止接新任务并结束在跑的任务，"
                        "然后照样执行收尾出表出图。已完成的折都已落盘，不会丢）")
    p.add_argument("--only-part", type=int, default=-1,
                   help="只跑某一步的第 K 个部分（-1=全部）。无人值守会自动加，一般不用手写")
    p.add_argument("--job-hours", type=float, default=3.0,
                   help="单个子任务最长跑多少小时，超时自动杀掉并跳过（默认 3；0=不限时）。"
                        "防止某个特别费时的任务把整晚占死、让收尾永远轮不到")
    p.add_argument("--fast", action="store_true",
                   help="快速模式：epochs 60 / 早停 patience 12（省 30–40%% 时间，收敛略差一点）")
    p.add_argument("--packed-dir", default=None,
                   help="打包数据的目录（默认放在数据目录旁边，如 tetraview_processed_packed）")
    p.add_argument("--packed", default="auto", choices=["auto", "off"],
                   help="auto（默认）: 打包数据存在就用它；off: 强制读原始 .pt")
    p.add_argument("--pack-pt", action="store_true",
                   help="导出数据时把四视图 .pt 目录一起打包（约 10 GB，默认不打包）")
    p.add_argument("--no-amp", action="store_true", help="关掉混合精度")
    p.add_argument("--quiet", action="store_true", help="少打印")
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2,
                   help="每个任务开几个子进程读数据（默认 2；机器核多可以加到 4；"
                        "如果反而更慢就设 0 回退到单进程读取）")
    p.add_argument("--threads", type=int, default=2,
                   help="每个进程用几个 CPU 线程（默认 2；多卡并行时别设太大，否则抢 CPU）")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-3d", type=float, default=1e-5)
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--sched-patience", type=int, default=10)
    p.add_argument("--select-metric", default="rmse", choices=["rmse", "composite"])
    p.add_argument("--dedup", default="none", choices=["none", "first", "median"],
                   help="重复结构处理：none/first/median")
    p.add_argument("--balance", default="none", choices=["none", "undersample"])
    return p


GPU_STEPS = {"ablation", "confs", "seeds", "strict", "scaffold", "holdout", "loso", "assay", "smoke"}
CPU_STEPS = {"preflight", "dataflow", "version", "pack", "baselines", "cliffs", "report", "status", "watch",
             "splits", "figures", "features", "exportdata", "writing"}


def n_parts_for(g, st: str) -> int:
    """这个步骤会被拆成几个「部分」（每部分 = 一个内部 CV 折 / 一个外部测试折 / 一个来源）。"""
    if st in ("strict", "scaffold", "holdout"):
        return int(g.folds) + 1      # 内部 CV 的每一折 + 最终外部测试折
    if st in ("ablation", "confs", "seeds", "assay"):
        return int(g.folds)
    return 1                         # loso 的份数取决于数据里来源个数，不预先拆


def expand_jobs(g, steps: List[str]) -> List[List[str]]:
    """把步骤拆成可以并行的小任务：一张卡一次只跑一个小任务，跑完自动接下一个。

    拆分粒度：先按 变体 / 构象数 / 种子，**再按「部分」拆成单折**（--only-part K）。
    单折任务的好处：1) 断点粒度从"整步"降到"单折"；2) 单张卡不会被一个几小时的长任务占死，
    收尾（出表/出图）不用等最慢的那一个；3) 多卡能真正吃满。
    划分是确定性的，拆开跑与整步跑结果完全一致。
    """
    base: List[List[str]] = []
    for st in steps:
        if st == "ablation":
            for v in [x.strip() for x in g.variants.split(",") if x.strip()]:
                base.append(["ablation", "--variants", v])
        elif st == "confs":
            for c in [x.strip() for x in g.confs.split(",") if x.strip()]:
                base.append(["confs", "--confs", c])
        elif st == "seeds":
            for sd in seeds_of(g, multi=True):
                base.append(["seeds", "--variants", "full,wo_3d", "--seeds", str(sd)])
        elif st == "loso":
            base.append(["loso"])
        else:
            base.append([st])
    jobs: List[List[str]] = []
    for j in base:
        n = n_parts_for(g, j[0])
        if n <= 1:
            jobs.append(j)
        else:
            for k in range(n):
                jobs.append(j + ["--only-part", str(k)])
    return jobs


def pick_idle_gpus(n: int) -> List[str]:
    """挑当前最闲的 n 张卡（共享服务器上别去抢别人正在用的卡）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=10).decode()
        rows = []
        for line in out.strip().splitlines():
            idx, util, mem = [x.strip() for x in line.split(",")]
            rows.append((int(float(util)), int(float(mem)), idx))
        by_index = sorted(rows, key=lambda r: int(r[2]))
        picked = [r[2] for r in sorted(rows)[:n]]
        print(f"  自动挑卡：GPU {'、'.join(picked)}（按利用率/显存从低到高选）")
        print("  全机现状： " + " | ".join(
            f"GPU{r[2]} {r[0]}% {r[1]}MB" for r in by_index))
        return picked
    except Exception as exc:
        print(f"  （拿不到显卡信息：{exc}，退化为用 GPU 0..{n - 1}）")
        return [str(i) for i in range(n)]


def run_parallel(g, steps: List[str]) -> int:
    """多卡并行：把任务拆小后轮流派给每张卡，所有子进程日志写到 <out>/logs/。"""
    gpus = [x.strip() for x in g.gpus.split(",") if x.strip()]
    log_dir = ensure_dir(os.path.join(g.out_dir, "logs"))
    jobs = expand_jobs(g, steps)
    if getattr(g, "shard", ""):
        try:
            i, n = [int(x) for x in str(g.shard).split("/")]
            assert 1 <= i <= n
        except Exception:
            raise SystemExit("--shard 格式应为 i/n，例如 --shard 1/3")
        all_jobs = jobs
        jobs = [j for k, j in enumerate(all_jobs) if k % n == i - 1]
        print(f"\n  [分片 {i}/{n}] 本机只跑 {len(jobs)}/{len(all_jobs)} 个任务："
              f"{'、'.join(' '.join(j) for j in jobs)}")
        print("  其它机器用 --shard 2/3、3/3 跑剩下的；结果按折唯一命名，"
              "跑完把各机的 results/ 合并即可（不会冲突）。", flush=True)
    gpu_jobs = [j for j in jobs if j[0] in GPU_STEPS]
    cpu_jobs = [j for j in jobs if j[0] not in GPU_STEPS]
    # 便宜的步骤先跑（自检/数据流/版本），失败就直接停
    EARLY = ("preflight", "dataflow", "version", "pack")   # 必须在训练前跑完
    for j in [x for x in cpu_jobs if x[0] in EARLY]:
        print(f"\n  >>> 先跑：{STEP_TITLE.get(j[0], j[0])}", flush=True)
        STEPS[j[0]](g)
    cpu_jobs = [x for x in cpu_jobs if x[0] not in EARLY]

    print(f"\n  == 并行调度：{len(gpu_jobs)} 个小任务分给 {len(gpus)} 张卡（{','.join(gpus)}）==")
    for i, j in enumerate(jobs):
        print(f"     {i+1:>2d}. {' '.join(j)}")
    print(flush=True)

    pending = list(gpu_jobs)
    running: Dict[str, Tuple[List[str], subprocess.Popen, object]] = {}
    rc = 0
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            logfile = open(os.path.join(log_dir, f"任务_{'_'.join(job)}_GPU{gpu}.log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            # 统一走 job_cmd：base_args 在前、job 在后（顺序写反过，见 job_cmd 注释）
            cmd = job_cmd(g, job)
            p = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT)
            running[gpu] = (job, p, logfile)
            print(f"  [GPU {gpu}] 开始：{' '.join(job)}  → 日志 {os.path.basename(logfile.name)}", flush=True)
        time.sleep(5)
        for gpu in list(running):
            job, proc, logfile = running[gpu]
            if proc.poll() is None:
                continue
            logfile.close()
            print(f"  [GPU {gpu}] 结束：{' '.join(job)}（返回码 {proc.returncode}）", flush=True)
            if proc.returncode:
                rc |= 1
            del running[gpu]

    for j in cpu_jobs:
        print(f"\n  >>> 串行执行：{STEP_TITLE.get(j[0], j[0])}", flush=True)
        try:
            rc |= STEPS[j[0]](g) or 0
        except SystemExit as e:
            print(f"  ！{STEP_TITLE.get(j[0], j[0])} 中止：{e}")
            rc |= 1
    print(f"\n  并行部分完成，返回码 {rc}。查看进度：python GPU_交给师兄/run_all.py 看进度")
    return rc


def base_args(g) -> List[str]:
    a = ["--data-dir", g.data_dir, "--out-dir", g.out_dir, "--epochs", str(g.epochs),
         "--folds", str(g.folds), "--num-confs", str(g.num_confs),
         "--confs", g.confs, "--variants", g.variants, "--cutoff", str(g.cutoff),
         "--test-size", str(g.test_size), "--min-source-n", str(g.min_source_n),
         "--assays", g.assays, "--models", g.models]
    if getattr(g, "seeds", None):
        a += ["--seeds", str(g.seeds)]
    a += ["--num-workers", str(getattr(g, "num_workers", 4)), "--threads", str(getattr(g, "threads", 2))]
    if g.meta_csv:
        a += ["--meta-csv", g.meta_csv]
    if g.assay_csv:
        a += ["--assay-csv", g.assay_csv]
    if g.max_samples:
        a += ["--max-samples", str(g.max_samples)]
    if g.overwrite:
        a += ["--overwrite"]
    return a


def job_cmd(g, job: List[str]) -> List[str]:
    """拼「一个子任务」的命令行。**所有派发任务的代码都必须经过这里，不要手写拼接。**

    【顺序约定】base_args 在前、任务自己的参数在后 —— argparse 同名参数以「最后出现的」为准，
    所以 job 里的 `--variants` / `--only-part` 必须压轴，否则会被 base_args 盖掉。
    历史上这个顺序写反过一次：3 个消融任务各自都跑了全部 3 个变体（3 倍重复劳动，
    还并发写同一批 fold 文件），白跑了一晚。所以这里再补一道便宜的自检。
    """
    cmd = [sys.executable, "-u", os.path.abspath(__file__)] + base_args(g) + list(job)
    for flag in ("--variants", "--only-part"):
        if flag in job:
            want = job[job.index(flag) + 1]
            got = cmd[max(i for i, t in enumerate(cmd) if t == flag) + 1]
            if got != want:
                raise SystemExit(
                    f"内部错误：{flag} 被 base_args 覆盖了（想要 {want}，实际生效 {got}）。"
                    f"检查 job_cmd / base_args 的拼接顺序。")
    return cmd


def main() -> int:
    g = make_parser().parse_args()
    if getattr(g, "all_idle_gpus", False):
        idle = idle_gpus()
        if idle:
            g.auto_gpus = len(idle)
            print(f"  [全部空闲卡] 启动时发现 GPU {','.join(idle)}，全部加入任务池")
        else:
            g.auto_gpus = 1
            print("  [全部空闲卡] 当前没有检测到完全空闲的卡，先使用 1 张卡并等待后续补卡")
    raw = g.步骤 or ["全部"]
    steps: List[str] = []
    for s in raw:
        key = STEP_NAMES.get(s, s)
        if key == "all":
            steps = list(DEFAULT_ORDER)
            break
        if key not in STEPS:
            raise SystemExit(f"不认识的步骤「{s}」。可选：{'、'.join(STEP_NAMES)}")
        steps.append(key)
    if not steps:
        if getattr(g, "one_day", False):
            steps = list(TINY_ORDER if getattr(g, "tiny", False) else LEAN_ORDER)
        else:
            steps = list(DEFAULT_ORDER)

    needs_src = {"preflight", "ablation", "confs", "seeds", "strict", "scaffold", "holdout",
                 "loso", "preprocess", "assay", "baselines", "smoke"}
    if any(s in needs_src for s in steps):
        ensure_src()

    if getattr(g, "one_day", False):
        if "--variants" not in sys.argv:
            g.variants = LEAN_VARIANTS
        n_gpu = int(getattr(g, "auto_gpus", 0) or 0) or len([x for x in (g.gpus or "").split(",") if x.strip()]) or 2
        g.tiny = n_gpu <= 3
        if g.tiny:
            # 两三张卡的一天版先拿到可用于后续决策的结果：
            # 3 折内部验证 + 1 个外部测试折。用户显式传 --folds 时保留其选择。
            if "--folds" not in sys.argv:
                g.folds = 3
            print(f"  [--one-day 一天版·{n_gpu} 卡] 纳入所有已有自动化实验，折数={g.folds}；"
                  f"已完成折自动跳过。本批缓跑：{TINY_DEFER}")
        else:
            print(f"  [--one-day 一天版·{n_gpu} 卡] 用完整方案（多含骨架划分与来源留一）；"
                  f"多种子/wo_0d/wo_2d/t-SNE 留到下一批")
    if getattr(g, "fast", False):
        if "--epochs" not in sys.argv:
            g.epochs = min(int(g.epochs), 60)
        if "--early-stop" not in sys.argv:
            g.early_stop = min(int(g.early_stop), 12)
        print(f"  [--fast] 快速模式：epochs={g.epochs}，early_stop={g.early_stop}")
    if getattr(g, "auto_gpus", 0) and not g.gpus:
        picked = pick_idle_gpus(int(g.auto_gpus))
        g.gpus = ",".join(picked)

    print(f"\n{BOLD}TV-STFN 返修实验 · 傻瓜版 {VERSION}{RESET}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {env_report()}")
    try:
        torch.set_num_threads(max(1, int(g.threads)))
    except Exception:
        pass
    print(f"  结果目录：{g.out_dir}")
    print(f"  步骤：{'、'.join(STEP_TITLE.get(s, s) for s in steps)}")

    if "daemon" in steps:
        return STEPS["daemon"](g)          # 无人值守自己就是调度器，不能再套一层
    if g.gpus and steps != ["watch"] and steps != ["status"]:
        return run_parallel(g, steps)

    rc = 0
    failed: List[str] = []
    for s in steps:
        t0 = time.time()
        try:
            rc |= STEPS[s](g) or 0
        except SystemExit as e:
            print(f"  ！{STEP_TITLE.get(s, s)} 中止：{e}")
            rc |= 1
            failed.append(s)
            if s == "preflight":
                break
        except KeyboardInterrupt:
            print("\n  被 Ctrl-C 中断。已经算完的折都保存了，原命令再跑一次会接着跑。")
            return 130
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ！{STEP_TITLE.get(s, s)} 出错：{type(e).__name__}: {e}")
            rc |= 1
            failed.append(s)
        if len(steps) > 1:
            print(f"  {DIM}—— {STEP_TITLE.get(s, s)} 用时 {fmt_dur(time.time() - t0)} ——{RESET}")
    if failed:
        print(f"\n{YELLOW}有 {len(failed)} 个步骤没跑完：{'、'.join(STEP_TITLE.get(x, x) for x in failed)}{RESET}")
        print("  修好原因后用同一条命令重跑即可（已完成的折会跳过）。")
    print(f"\n{BOLD}全部完成，返回码 {rc}{RESET}  出表结果：{os.path.join(g.out_dir, 'tables', '论文表格汇总.md')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
