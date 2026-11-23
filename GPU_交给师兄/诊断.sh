#!/usr/bin/env bash
# ============================================================================
# 60 秒诊断 v3：任务到底在跑、在挂着、还是早就失败了？
#
# 用法（在服务器上，TV-STFN 项目根目录里）：
#     bash GPU_交给师兄/诊断.sh
#
# 只读不写：不启动任务、不 kill 任何东西、不改任何文件。
# 把整段输出发我即可。
#
# v3 改动（2026-09-20 15:30）：
#   日志文件不再写死 `看守.log`。看守输出可能落在三个地方，本脚本自动挑
#   最新改动的那个，并在开头明确打印「我读的是哪个文件」：
#       GPU_交给师兄/看守.log        —— 守护.sh / 重启_6小时版.sh(v2+) 的看守日志
#       GPU_交给师兄/守护.out        —— 重启_6小时版.sh(v1) 直接把 nohup 输出写这里
#       重启_6小时版.txt             —— 师兄用 nohup 重定向到项目根时落在这里
#   起因：师兄 v1 脚本那次输出全在 守护.out，而本脚本只读 看守.log，
#   于是所有小节都显示「(没有 看守.log)」，看起来像“什么都没跑” —— 其实一直在跑。
# ============================================================================
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="$REPO/tvstfn_paper_pipeline/outputs/reviewer_gpu_followup"
NOW=$(date '+%F %T')

# ---- 自动挑「最新改动」的看守日志 ----
LOGF=""
for c in "$HERE/看守.log" "$HERE/守护.out" "$REPO/重启_6小时版.txt" "$REPO/守护.out"; do
  [ -f "$c" ] || continue
  if [ -z "$LOGF" ] || [ "$c" -nt "$LOGF" ]; then LOGF="$c"; fi
done
[ -z "$LOGF" ] && LOGF="$HERE/看守.log"     # 一个都没有：保留旧名，让下面照常报「没有」

hr() { printf '\n%s\n' "=========== $* ==========="; }

hr "0. 现在几点 / 在哪 / 我在读哪个日志"
echo "现在：$NOW  主机：$(hostname)"
echo "项目根：$REPO"
if [ -f "$LOGF" ]; then
  echo "看守日志：$LOGF"
  echo "          （最后改动 $(stat -c '%y' "$LOGF" 2>/dev/null || stat -f '%Sm' "$LOGF" 2>/dev/null)）"
else
  echo "看守日志：$LOGF  ← ⚠️ 不存在"
fi

hr "1. 进程（有没有活的）"
if ps -ef | grep -E "守护\.sh|run_all\.py" | grep -v grep; then
  echo "→ 有进程。注意看它已经跑了多久（ELAPSED 那一列），以及命令行参数。"
  echo "  要盯进度（另开终端，不影响它）："
  echo "      tail -f $LOGF"
else
  echo "→ ⚠️ 没有任何 守护.sh / run_all.py 进程：任务已经停了。"
fi

hr "2. 启动时的计划：几张卡、几个任务（这是解读后面一切的前提）"
if [ -f "$LOGF" ]; then
  grep -nE "启动|计划执行|工作量估算|空闲卡|自动挑卡|待跑任务|先跑：|一天版|使用显卡" "$LOGF" | head -30
else
  echo "(没有 $LOGF)"
fi

hr "3. 所有 开始 / 完成 / 失败 / 跳过 事件（最重要的 40 行）"
if [ -f "$LOGF" ]; then
  grep -nE "开始：|完成：|失败|跳过|发现空卡|看守退出|无人值守结束" "$LOGF" | tail -40
  echo
  echo "-- 计数 --"
  echo "开始：  $(grep -c '开始：' "$LOGF" 2>/dev/null)"
  echo "完成：  $(grep -c '完成：' "$LOGF" 2>/dev/null)"
  echo "失败：  $(grep -c '失败（rc=' "$LOGF" 2>/dev/null)"
  echo "跳过：  $(grep -c '连续失败' "$LOGF" 2>/dev/null)"
else
  echo "(没有 $LOGF)"
fi

hr "4. 心跳的最后 3 条 + 心跳停止了多久"
if [ -f "$LOGF" ]; then
  grep "心跳：" "$LOGF" | tail -3
  last=$(grep "心跳：" "$LOGF" | tail -1 | sed -n "s/^\[\([^]]*\)\].*/\1/p")
  echo "最后一条心跳时间戳：${last:-未知}   现在：$NOW"
  echo "（注意：心跳只是看守自己在报，心跳在走 ≠ 任务在算；要看第 6 节任务日志的 mtime）"
else
  echo "(没有 $LOGF)"
fi

hr "5. 显卡现状"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
else
  echo "(没有 nvidia-smi)"
fi

hr "6. 任务日志清单（按最近修改倒序）"
if [ -d "$OUT/logs" ]; then
  ls -lt "$OUT/logs"/任务_*.log 2>/dev/null | head -10
  echo
  new=$(ls -t "$OUT/logs"/任务_*.log 2>/dev/null | head -1)
  if [ -n "$new" ]; then
    echo "→ 最新那个是：$(basename "$new")"
    echo "  它的最后修改时间：$(stat -c '%y' "$new" 2>/dev/null || stat -f '%Sm' "$new")"
    echo "  （若这个时间比现在早很多 → 这个任务卡死了，不是在算）"
    echo
    echo "  最后 25 行："
    tail -n 25 "$new"
  fi
else
  echo "(没有 $OUT/logs。如果日志里有 nohup 一行，说明脚本压根没进到这一步)"
fi

hr "7. 已完成多少折"
if [ -d "$OUT/results" ]; then
  find "$OUT/results" -path "*/folds/*.json" 2>/dev/null | while read -r f; do echo "$f"; done > /tmp/_foldlist.$$
  echo "合计：$(wc -l < /tmp/_foldlist.$$) 折"
  find "$OUT/results" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | while read -r d; do
    c=$(find "$d" -path "*/folds/*.json" 2>/dev/null | wc -l)
    printf '   %-28s %s 折\n' "$(basename "$d")" "$c"
  done
  rm -f /tmp/_foldlist.$$
else
  echo "(还没有 results 目录 → 一折都没落盘)"
fi

hr "8. 汇总表 / 收尾产物"
for f in tables/论文表格汇总.md splits figures features; do
  if [ -e "$OUT/$f" ]; then echo "  有  $f"; else echo "  无  $f"; fi
done

hr "9. 状态快照 json"
[ -f "$OUT/logs/看守状态.json" ] && cat "$OUT/logs/看守状态.json" || echo "(无)"

hr "10. 解读口诀"
cat <<'EOF'
  · 命令：进程在 + 最新任务日志的 mtime 在往前走  → 正常在算，等着。
  · 卡死：进程在 + 最新任务日志 mtime 停在几小时前 → 那个任务挂了。
          心跳还会继续打（心跳是看守自己的，不代表任务在算）。
  · 全灭：失败计数 >> 完成计数                       → 一批任务在崩，看第 6 节日志的报错。
  · 没起：没有进程 + 看守日志为空                    → 原命令重跑一遍即可（断点续跑）。
  不要做的事：不要 kill 正在训练的子进程；不要起第二个守护。
EOF

hr "11. ⚠️ 本脚本是【只读】的！跑一百遍结果都一样"
cat <<'EOF'
  诊断.sh 只打印信息：不启动任务、不 kill 进程、不改任何文件。
  所以「再跑一次诊断、结果没变」= 什么都没做，不是「修了没用」。

  真要动手（停旧 + 启新），跑这个（它会先核对代码版本）：
      bash GPU_交给师兄/重启_6小时版.sh

  怎么确认真的生效了：跑完之后，`ps -ef | grep run_all.py` 里应该出现
      run_all.py 无人值守 ... --gpus 0,1,2,3,7 ...
  而不是原来的
      run_all.py 无人值守 --all-idle-gpus --one-day

  ⚠️ 如果现在已经在跑了（第 1 节有进程、且卡号正确），就【不要再跑重启脚本】，
     重跑会把它停掉再从头接 —— 只会白白浪费几分钟。
EOF
