"""将配置里的 manifest 路径解析为仓库内存在的单个 .jsonl 文件。

约定：开发集 / 测试集清单均放在 ``data/manifests/train/`` 下；配置可写
``manifests_train_dir`` + **仅文件名**（如 ``dev_clean.jsonl``），或写相对/绝对路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_manifest_jsonl(repo_root: Path, ref: str, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or {}
    data = cfg.get("data") or {}
    train_dir = str(data.get("manifests_train_dir", "data/manifests/train")).strip()
    ref = str(ref).replace("${REPO}", str(repo_root)).strip()
    p = Path(ref)
    if p.is_file():
        return p.resolve()
    p2 = (repo_root / ref).resolve()
    if p2.is_file():
        return p2
    base = (repo_root / train_dir).resolve()
    name = Path(ref).name
    p3 = (base / name).resolve()
    if p3.is_file():
        return p3
    raise FileNotFoundError(
        f"manifest not found: {ref!r} (tried {p}, {p2}, {p3}; manifests_train_dir={train_dir!r})"
    )
