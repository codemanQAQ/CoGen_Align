#!/usr/bin/env python3
"""Stage 2 generative alignment — scaffold; extend with DDP, WER eval (see doc)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cogen_align.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load merged YAML and print key training fields only.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _ROOT / args.config
    cfg = load_config(cfg_path)

    if args.dry_run:
        print("run_name:", cfg.get("run_name"))
        print("output_dir:", cfg.get("output_dir"))
        print("warmup_init_path:", cfg.get("training", {}).get("warmup_init_path"))
        print("train_manifest:", cfg.get("data", {}).get("train_manifest"))
        return

    raise SystemExit(
        "Full Stage2 training not wired in scaffold. Use --dry-run or implement "
        "loop per doc/CoGen_Align_Engineering_Handbook.docx (PEFT + CE + WER)."
    )


if __name__ == "__main__":
    main()
