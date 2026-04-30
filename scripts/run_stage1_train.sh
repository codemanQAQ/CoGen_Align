#!/usr/bin/env bash
# Stage1 对比学习 Warmup（无需 Stage2；Stage2 的 projector 初始化依赖本阶段产出的 ckpt_last.pt）
#
# 用法（在任意目录）：
#   bash /path/to/cogen-align/scripts/run_stage1_train.sh
#   bash .../run_stage1_train.sh configs/stage1/default.yaml
# 环境变量：
#   CONFIG        相对仓库根的 yaml，默认 configs/stage1/table1_s1_50h.yaml
#   NPROC         torchrun 本地进程数（通常=GPU 数），默认 1
#   MASTER_ADDR / MASTER_PORT  多卡时 torchrun 默认会设；单机多卡可显式：
#                  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501
#   WANDB_DISABLED 设为 1 不写 wandb（默认不设则按 yaml 尝试登录 wandb）
#
# 训练结束后在 yaml 的 output_dir 下会生成：
#   experiment_record.yaml / experiment_config.yaml / experiment_record.jsonl（流程与配置快照）
#   metrics.jsonl（每 log_every 步 train 标量 + eval 时 val 检索指标，与 WANDB_DISABLED 无关，便于离线绑图）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-${CONFIG:-configs/stage1/table1_s1_50h.yaml}}"
NPROC="${NPROC:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "config not found: ${REPO_ROOT}/${CONFIG}" >&2
  exit 1
fi

echo "[run_stage1] repo=${REPO_ROOT} config=${CONFIG} nproc=${NPROC} master=${MASTER_ADDR}:${MASTER_PORT}"
exec torchrun --nproc_per_node="${NPROC}" scripts/train_stage1.py --config "${CONFIG}"
