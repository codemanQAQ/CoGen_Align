#!/usr/bin/env bash
# =============================================================================
# 按数据量从小到大分批跑：每个小时档依次
#   (1) Baseline：仅 Stage2 生成式对齐（无 warmup）
#   (2) CoGen：    Stage1 对比 warmup → Stage2（带 ckpt_last 初始化）
#
# 无命令行参数。环境变量：
#   NPROC              torchrun 进程数，默认 1
#   MASTER_ADDR/PORT   默认 127.0.0.1 / 29501
#   WANDB_DISABLED     可选
#   SKIP_STAGE2=1      跳过所有 Stage2（当前 train_stage2 仍为 scaffold 时可先只跑 Stage1）
#
# 数据量顺序：50 → 100 → 200 → 300 → 400 → 500（小时）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"

_HOURS=(50 100 200 300 400 500)

_stage1_config() {
  local h="$1"
  case "${h}" in
    50)  echo "configs/stage1/table1_s1_50h.yaml" ;;
    100) echo "configs/stage1/default.yaml" ;;
    200|300|400|500) echo "configs/stage1/table1_s1_${h}h.yaml" ;;
    *) echo >&2 "invalid hour: ${h}"; exit 1 ;;
  esac
}

_torch() {
  torchrun --nproc_per_node="${NPROC}" "$@"
}

echo "[batch] repo=${REPO_ROOT} NPROC=${NPROC} SKIP_STAGE2=${SKIP_STAGE2}"

for h in "${_HOURS[@]}"; do
  echo ""
  echo "######################################################################"
  echo "# 数据量 ${h}h — Baseline（Stage2 only）"
  echo "######################################################################"
  if [[ "${SKIP_STAGE2}" == "1" ]]; then
    echo "[batch] SKIP_STAGE2=1，跳过 Baseline Stage2"
  else
    _torch scripts/train_stage2.py --config "configs/stage2/baseline_${h}h.yaml"
  fi

  echo ""
  echo "######################################################################"
  echo "# 数据量 ${h}h — CoGen Stage1（warmup）"
  echo "######################################################################"
  _torch scripts/train_stage1.py --config "$(_stage1_config "${h}")"

  echo ""
  echo "######################################################################"
  echo "# 数据量 ${h}h — CoGen Stage2（读取 Stage1 ckpt_last.pt）"
  echo "######################################################################"
  if [[ "${SKIP_STAGE2}" == "1" ]]; then
    echo "[batch] SKIP_STAGE2=1，跳过 CoGen Stage2"
  else
    _torch scripts/train_stage2.py --config "configs/stage2/cogen_${h}h.yaml"
  fi
done

echo ""
echo "[batch] 全部数据量档（50–500h）的 Baseline+CoGen 流水线已按序结束。"
