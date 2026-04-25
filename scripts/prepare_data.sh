#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Prepare LibriSpeech manifests under ${ROOT}/data/manifests (jsonl)."
echo "See data/manifests/README.md for schema. No downloads performed here."
