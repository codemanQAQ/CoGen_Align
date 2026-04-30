#!/usr/bin/env bash
# 遍历仓库 data/manifests 下全部 .jsonl，在原路径原地写入 feature_path（先写临时文件再覆盖）。
# 使用方式（在任意目录执行均可）：
#   bash /path/to/cogen-align/scripts/get_manifests/attach_all_manifests_features.sh
# 可选环境变量：
#   CONFIG   默认 configs/base.yaml（相对仓库根）
#   CHECK_NPY 设为 1 则加 --check-npy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/base.yaml}"
ATTACH_PY="${SCRIPT_DIR}/attach_feature_paths.py"
MAN_ROOT="${REPO_ROOT}/data/manifests"

if [[ ! -f "${ATTACH_PY}" ]]; then
  echo "missing ${ATTACH_PY}" >&2
  exit 1
fi

mapfile -t files < <(find "${MAN_ROOT}" -type f -name '*.jsonl' | sort)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no jsonl under ${MAN_ROOT}" >&2
  exit 1
fi

for f in "${files[@]}"; do
  tmp="${f}.tmp.$$"
  rm -f "${tmp}"
  echo "==> in-place ${f}"
  set +e
  if [[ "${CHECK_NPY:-0}" == "1" ]]; then
    python "${ATTACH_PY}" \
      --config "${CONFIG}" \
      --manifest-in "${f}" \
      --manifest-out "${tmp}" \
      --check-npy
  else
    python "${ATTACH_PY}" \
      --config "${CONFIG}" \
      --manifest-in "${f}" \
      --manifest-out "${tmp}"
  fi
  ec=$?
  set -e
  if [[ "${ec}" -ne 0 ]]; then
    rm -f "${tmp}"
    echo "failed: ${f}" >&2
    exit "${ec}"
  fi
  mv -f "${tmp}" "${f}"
done

echo "[OK] all manifests updated in place under ${MAN_ROOT}"
