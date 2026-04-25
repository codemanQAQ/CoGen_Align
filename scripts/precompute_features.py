#!/usr/bin/env python3
"""Precompute Whisper encoder features -> .npy paths referenced by manifests."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--manifest-in", type=str, required=True)
    p.add_argument("--manifest-out", type=str, required=True)
    args = p.parse_args()
    raise SystemExit(
        "TODO: load Whisper, iterate manifest-in jsonl, write feature .npy under "
        f"data/features and emit manifest-out with feature_path. Config: {args.config}"
    )


if __name__ == "__main__":
    main()
