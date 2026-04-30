"""Write reproducibility / workflow metadata next to training outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _git_head(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_status_porcelain(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or "clean"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def write_experiment_record(
    out_dir: str | Path,
    *,
    cfg: dict[str, Any],
    config_path: Path,
    argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """
    在 ``out_dir`` 写入 ``experiment_record.yaml``（流程记录）与 ``experiment_config.yaml``（合并后配置副本）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = repo_root or Path.cwd()

    meta: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": argv if argv is not None else sys.argv,
        "config_path": str(config_path.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "git_commit": _git_head(root),
        "git_status_porcelain": _git_status_porcelain(root),
    }
    if extra:
        meta["extra"] = extra

    rec_path = out / "experiment_record.yaml"
    with open(rec_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    cfg_path = out / "experiment_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # 机器可读一行摘要，便于流水线追加
    line = {
        "ts": meta["started_at_utc"],
        "config": str(config_path),
        "out_dir": str(out.resolve()),
        "world_size": meta["world_size"],
        "git": meta["git_commit"],
    }
    with open(out / "experiment_record.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

    return rec_path
