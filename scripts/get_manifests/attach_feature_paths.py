#!/usr/bin/env python3
"""
只读 manifest（jsonl），按 audio_path 与目录根规则插入 feature_path，不写 .npy。

特征路径规则与预计算脚本一致：见 cogen_align.data.feature_paths.audio_path_to_feature_path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cogen_align.data.feature_paths import audio_path_to_feature_path  # noqa: E402
from cogen_align.utils.config import load_config  # noqa: E402


def iter_manifest_jsonl(manifest_path: Path) -> Iterator[dict[str, Any]]:
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def attach_feature_path(
    row: dict[str, Any],
    *,
    audio_root: Path,
    feature_root: Path,
) -> dict[str, Any]:
    """复制一行并写入 feature_path（绝对路径）。"""
    audio_path = Path(row["audio_path"]).expanduser().resolve()
    feat = audio_path_to_feature_path(
        audio_path, audio_root=audio_root, feature_root=feature_root
    )
    out = dict(row)
    out["feature_path"] = str(feat.resolve())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs/base.yaml"),
        help="可选：从 YAML 读取默认 audio_root / feature_root",
    )
    p.add_argument("--manifest-in", type=str, required=True)
    p.add_argument("--manifest-out", type=str, required=True)
    p.add_argument("--audio-root", type=str, default=None)
    p.add_argument("--feature-root", type=str, default=None)
    p.add_argument(
        "--check-npy",
        action="store_true",
        help="若设置，则对每一行检查 .npy 是否存在；缺失则打印警告（仍写出 manifest）",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _REPO_ROOT / args.config
    cfg = load_config(cfg_path)
    data_cfg = cfg.get("data", {})

    audio_root = Path(
        args.audio_root or data_cfg.get("audio_root", "")
    ).expanduser().resolve()
    feature_root = Path(
        args.feature_root or data_cfg.get("feature_root", "")
    ).expanduser().resolve()

    manifest_in = Path(args.manifest_in).expanduser().resolve()
    manifest_out = Path(args.manifest_out).expanduser().resolve()
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        n_lines = sum(1 for _ in open(manifest_in, encoding="utf-8") if _.strip())
    except OSError:
        n_lines = None

    missing = 0
    with open(manifest_out, "w", encoding="utf-8") as fout:
        for row in tqdm(
            iter_manifest_jsonl(manifest_in),
            total=n_lines,
            desc="attach feature_path",
            ncols=100,
        ):
            out = attach_feature_path(row, audio_root=audio_root, feature_root=feature_root)
            if args.check_npy and not Path(out["feature_path"]).is_file():
                missing += 1
                if missing <= 20:
                    print(f"[Warn] missing npy: {out['feature_path']}")
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {manifest_out}")
    if args.check_npy:
        print(f"[OK] missing_npy_count={missing}")


if __name__ == "__main__":
    main()
