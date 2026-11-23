#!/usr/bin/env bash
# ============================================================================
# 6 小时版：停掉旧看守 → 用指定显卡跑「严格划分 + 消融 + 分assay」→
#           跑完自动出表 / 出图 / 导出划分 / 写作映射。
#
#   bash GPU_交给师兄/重启_6小时版.sh              # 默认 GPU 0,1,2,3,7 跑「严格 消融」
#   bash GPU_交给师兄/重启_6小时版.sh 0,1,2,3,5,6,7 严格 消融 分assay   # 5/6 空出来时加跑分assay
#   bash GPU_交给师兄/重启_6小时版.sh stop         # 只停，不启动
#
# 【重要】诊断.sh 是只读的，跑一百遍结果都一样；真正动手要跑本脚本。
# 怎么确认生效：跑完后 `ps -ef | grep run_all.py` 里应出现
#   run_all.py 无人值守 ... --gpus 0,1,2,3,7 ...
# 而不是旧的
#   run_all.py 无人值守 --all-idle-gpus --one-day
#
# v2 改动（2026-09-20 15:30）——统一日志文件名：
#   v1 把看守输出写进 `守护.out`，但脚本自己打印的提示、以及 诊断.sh 读的都是
#   `看守.log`，于是 `head GPU_交给师兄/看守.log` 会报 No such file —— 明明在跑，
#   看起来却像没跑。现在：看守输出统一写 `看守.log`，并额外把 `守护.out`
#   做成指向 `看守.log` 的软链，两个名字都 tail 得到。
#
# 设计要点（都是被上一次 45 小时坑过之后加的）：
#   · 先核对 run_all.py 版本，不是 2026-09-20.6 直接拒绝启动 —— 免得又跑一晚旧代码。
#   · 每个子任务只跑「一折」（--only-part），单任务最长 ~1.5 小时：
#     一张卡不会被长任务占死，收尾不用等最慢的那个，中断粒度降到单折。
#   · 单任务超时 --job-hours 3：超时杀掉并跳过，原样记进 logs/失败记录.md。
#   · 总时长硬上限 --max-hours 5：到点结束在跑的任务并**照常执行收尾**，
#     保证 6 小时内一定拿到出表/出图，不会再被某个任务堵死。
#   · 失败不再"只写一句请检查"：报错尾巴会自动打进看守日志和 logs/失败记录.md。
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
LOG="$HERE/看守.log"
LOG_DIR="$REPO/tvstfn_paper_pipeline/outputs/reviewer_gpu_followup/logs"
# 门禁：接受 2026-09-20 的 .4 及以上（.4 修了消融必崩、.5 加变体自检、.6 修另一条派发路径）
# 用正则而不是写死单个版本号，这样你覆盖到 .5 或 .6 都能过。
NEED_VERSION_RE="2026-09-20\.[4-9]"
NEED_VERSION_HINT="2026-09-20.5 或 .6"

GPULIST="${1:-0,1,2,3,7}"
if [ "$GPULIST" = "stop" ]; then shift; GPULIST=""; else shift; fi
STEPS="${*:-严格 消融}"
JOB_HOURS=3
MAX_HOURS=5
cd "$REPO" || { echo "找不到项目根目录 $REPO"; exit 1; }

echo "=============================================================="
echo "项目根目录 : $REPO"
echo "使用显卡   : ${GPULIST:-（只停不启动）}"
echo "步骤       : $STEPS"
echo "单任务上限 : ${JOB_HOURS} 小时   总时长上限：${MAX_HOURS} 小时（到点自动收尾）"
echo "=============================================================="

# ---- 防手滑：已经在跑的时候别再重跑一遍（会停掉再从头接，白等几分钟）----
if ps -ef | grep "[r]un_all.py 无人值守" >/dev/null 2>&1; then
  echo
  echo "  ⚠️ 检测到已经有「run_all.py 无人值守」在跑。"
  echo "     本脚本会先把它【停掉】再重启 —— 如果你只是想看进度，请按 Ctrl-C。"
  echo "     看进度：tail -f GPU_交给师兄/看守.log"
  echo "     真要继续（停旧+启新）：5 秒后自动进行 …"
  sleep 5
fi

echo
echo "--- 第 0 步：核对代码版本（必须先过）---"
VER_OUT="$(python "$HERE/run_all.py" 版本 2>&1)"
echo "$VER_OUT"
if ! printf '%s' "$VER_OUT" | grep -qE "$NEED_VERSION_RE"; then
  echo
  echo "  ✗ 服务器上的 run_all.py 太旧（需要 $NEED_VERSION_HINT，实测不是）。"
  echo "    → 先把新版 run_all.py 覆盖上去再跑这个脚本（覆盖正在跑的 python 进程是安全的，"
  echo "      但为稳妥起见，本脚本会先把旧任务停掉）。"
  echo "    → 只想停旧任务就执行：bash GPU_交给师兄/重启_6小时版.sh stop"
  exit 1
fi
echo "  ✓ 版本正确（$NEED_VERSION_HINT 均可）"

echo
echo "--- 第 1 步：停掉旧的看守进程树 ---"
pkill -f "守护.sh" 2>/dev/null && echo "  已停：守护.sh（先停它，否则它会把 python 再拉起来）" || echo "  没有 守护.sh"
sleep 1
pkill -f "run_all.py 无人值守" 2>/dev/null && echo "  已停：无人值守调度器" || echo "  没有 无人值守"
pkill -f "run_all.py loso" 2>/dev/null && echo "  已停：loso（那个还要 117 小时的任务）" || echo "  没有 loso"
pkill -f "run_all.py 严格" 2>/dev/null || true
pkill -f "run_all.py 消融" 2>/dev/null || true
pkill -f "run_all.py 分assay" 2>/dev/null || true
sleep 5
echo
echo "--- 第 2 步：确认停干净 ---"
if ps -ef | grep -E "守护\.sh|run_all\.py" | grep -v grep; then
  echo "  还有残留，再等 5 秒："; sleep 5
  if ps -ef | grep -E "守护\.sh|run_all\.py" | grep -v grep; then
    echo "  ⚠️ 仍有残留，请把上面的 PID 发我，别急着启动新的。"
    exit 1
  fi
fi
echo "  已全部停止 ✓（已完成的折都在 results/ 里，不会丢）"

echo
echo "--- 显卡现状 ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv 2>/dev/null || true

if [ -z "$GPULIST" ]; then
  echo; echo "只停不启动，结束。"; exit 0
fi

echo
echo "--- 第 3 步：后台启动（可以关终端）---"
ARCH="$LOG.旧_$(date +%Y%m%d_%H%M)"
[ -f "$LOG" ] && mv "$LOG" "$ARCH" && echo "  旧日志已归档：$ARCH"
[ -f "$LOG_DIR/失败记录.md" ] && mv "$LOG_DIR/失败记录.md" "$LOG_DIR/失败记录_旧_$(date +%Y%m%d_%H%M).md"

# 看守输出统一写 看守.log；再给 守护.out 一个软链，方便老习惯两个名字都能 tail。
OUTF="$HERE/守护.out"
rm -f "$OUTF"
ln -s "看守.log" "$OUTF" 2>/dev/null && echo "  守护.out → 看守.log（软链，两个名字都能 tail）"

nohup bash -c '
  cd "$1" || exit 1
  for i in 1 2 3; do
    echo "[$(date "+%F %T")] ===== 看守第 $i 次启动（卡 $3，单任务上限 $5 小时，总上限 $6 小时）====="
    # 位置参数（步骤）必须在选项前面，和文档里的示例写法一致
    python -u "$2" 无人值守 $4 \
      --gpus "$3" --folds 5 --epochs 100 --variants full,wo_cliff,wo_3d \
      --job-hours "$5" --max-hours "$6"
    rc=$?
    echo "[$(date "+%F %T")] 看守退出，返回码 $rc"
    if [ "$rc" -eq 0 ]; then
      echo "[$(date "+%F %T")] 全部完成 ✓ 汇总表：tvstfn_paper_pipeline/outputs/reviewer_gpu_followup/tables/论文表格汇总.md"
      break
    fi
    echo "[$(date "+%F %T")] 20 秒后自动重试"
    sleep 20
  done
' _ "$REPO" "$HERE/run_all.py" "$GPULIST" "$STEPS" "$JOB_HOURS" "$MAX_HOURS" > "$LOG" 2>&1 &

sleep 5
echo
echo "=============================================================="
echo "  如果上面出现了「看守第 1 次启动 / 使用显卡 / 单任务上限」"
echo "  并且 ps 里是 --gpus $GPULIST 而不是 --all-idle-gpus，"
echo "  那么 —— 新版已经生效，这次是真在跑了。"
echo "=============================================================="
cat > "$HERE/已启动_标记.txt" <<MARK
新版已启动（run_all.py 2026-09-20.4 及以上；当前为 $NEED_VERSION_HINT，重启_6小时版.sh v2）
时间    ：$(date '+%F %T')
显卡    ：$GPULIST
步骤    ：$STEPS
单任务上限：${JOB_HOURS} 小时
总时长上限：${MAX_HOURS} 小时（到点强制收尾）
判断生效：ps -ef | grep run_all.py  里应有 --gpus $GPULIST
看进度  ：tail -f GPU_交给师兄/看守.log      （守护.out 是同一个文件的软链）
看任务数：grep 待跑任务 GPU_交给师兄/看守.log | tail -1
失败原因：cat $LOG_DIR/失败记录.md
MARK
echo "已把这次的计划写进 $HERE/已启动_标记.txt"
echo
echo "已启动。进程确认（关键看 --gpus 那段）："
ps -ef | grep "run_all.py" | grep -v grep || echo "  （没看到进程？把 $LOG 发我）"
echo
echo "=============================================================="
echo "盯进度（另开终端，不影响它）："
echo "  cd $REPO"
echo "  tail -f GPU_交给师兄/看守.log"            # 日志现在会自己打印失败原因
echo "  python GPU_交给师兄/run_all.py 看进度"
echo "  python GPU_交给师兄/run_all.py 状态"
echo "  cat tvstfn_paper_pipeline/outputs/reviewer_gpu_followup/logs/失败记录.md"
echo "=============================================================="
