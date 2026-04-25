#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
echo "Run ablations by iterating configs/stage1/*.yaml and configs/stage2/*.yaml as needed."
