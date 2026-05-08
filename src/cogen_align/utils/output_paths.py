"""训练脚本共用的 output_dir 解析（多 seed 时挂在 outputs/<seed>/ 下）。"""

from __future__ import annotations

import os
from pathlib import Path


def effective_seed_output_root(
    *,
    cli_seed_output_root: str | None,
    run_tag: str | None,
    seed: int,
) -> str | None:
    """
    未显式传 --seed-output-root 时，默认用 str(seed) 将产物写入 outputs/<seed>/...
    例外：带 --run-tag 时保持旧版「扁平目录 + 后缀」；或设置环境变量 COGEN_OUTPUTS_FLAT=1 关闭嵌套。
    """
    if cli_seed_output_root is not None:
        sr = cli_seed_output_root.strip()
        if sr:
            return sr
        # 显式传空串视为「未指定」，继续走默认/扁平规则
    if run_tag:
        return None
    flat = os.environ.get("COGEN_OUTPUTS_FLAT", "").strip().lower()
    if flat in ("1", "true", "yes"):
        return None
    return str(seed)


def rewrite_stage1_warmup_for_nested_outputs(
    warmup_path: str,
    seed_output_subdir: str,
) -> str:
    """
    YAML 常为 ``outputs/stage1/foo/ckpt_last.pt``；当 Stage1 已写入 ``outputs/<seed>/stage1/...`` 时对齐路径。
    支持 ``${REPO}/outputs/stage1/...``。
    """
    raw = str(warmup_path)
    tmp = raw.replace("\\", "/")
    token = f"outputs/{seed_output_subdir}/stage1/"
    if token in tmp:
        return raw
    if "outputs/stage1/" not in tmp:
        return raw
    new_tmp = tmp.replace("outputs/stage1/", token, 1)
    return new_tmp


def resolve_training_output_dir(
    repo_root: Path,
    cfg_output_dir: str,
    *,
    seed_output_root: str | None = None,
    run_tag: str | None = None,
) -> str:
    """
    - seed_output_root: outputs 下的单层目录名（如 ``42``），YAML 里 ``outputs/...`` 去掉
      ``outputs/`` 前缀后接在 ``outputs/<seed_output_root>/`` 下。
    - run_tag: 仅在未设置 seed_output_root 时，拼在配置里的 output_dir 字符串末尾（旧行为）。
      若同时设置了 seed_output_root，则 run_tag 被忽略（由 CLI 层避免同时传入）。
    """
    raw = str(cfg_output_dir).replace("${REPO}", str(repo_root))
    bpath = Path(raw)
    if not bpath.is_absolute():
        bpath = (repo_root / bpath).resolve()
    out_root = (repo_root / "outputs").resolve()
    if seed_output_root:
        sr = seed_output_root.strip()
        if not sr or sr in (".", ".."):
            raise ValueError(f"非法 seed_output_root: {seed_output_root!r}")
        rel = bpath.relative_to(out_root)
        return str((out_root / sr / rel).resolve())
    if run_tag:
        if Path(raw).is_absolute():
            return str((bpath.parent / f"{bpath.name}_{run_tag}").resolve())
        return f"{raw}_{run_tag}"
    return str(bpath)
