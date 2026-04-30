#!/usr/bin/env bash
# Stage1：按数据量从小到大依次全量训练（50h → 100h → 200h → 300h → 400h → 500h）。
# 本脚本只做这一件事，不接受命令行参数。
#
# 多卡时设置环境变量（可选）：
#   NPROC=8              默认 1
#   MASTER_PORT=29502    默认 29501，并行多任务时改端口
#   WANDB_DISABLED=1     不上传 W&B
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

CONFIGS=(
  "configs/stage1/table1_s1_50h.yaml"
  "configs/stage1/default.yaml"
  "configs/stage1/table1_s1_200h.yaml"
  "configs/stage1/table1_s1_300h.yaml"
  "configs/stage1/table1_s1_400h.yaml"
  "configs/stage1/table1_s1_500h.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "[run_stage1_full] 缺少配置: ${REPO_ROOT}/${cfg}" >&2
    exit 1
  fi
  echo "[run_stage1_full] ===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") 开始 ${cfg} ====="
  torchrun --nproc_per_node="${NPROC}" scripts/train_stage1.py --config "${cfg}"
  echo "[run_stage1_full] ===== 结束 ${cfg} ====="
done

echo "[run_stage1_full] 全部 6 档数据量训练已完成。"
