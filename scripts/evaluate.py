#!/usr/bin/env python3
"""Evaluate WER / VoiceBench — placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, default="wer", choices=["wer", "voicebench"])
    args = p.parse_args()
    raise SystemExit(f"TODO: implement {args.task} for ckpt {args.checkpoint}")


if __name__ == "__main__":
    main()
