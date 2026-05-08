"""训练脚本共用的可复现性设置（随机种子、CUDA 确定性、DataLoader worker）。"""

from __future__ import annotations

import os
import random
from collections.abc import Callable

import numpy as np
import torch


def set_training_reproducibility(seed: int) -> None:
    """
    在创建模型 / Dataset / DataLoader 之前调用。
    覆盖：Python random、NumPy、PyTorch CPU&GPU、cudnn 确定性、
    CUBLAS 工作区（利于部分 matmul 可复现）。
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    s = int(seed)
    random.seed(s)
    np.random.seed(s % (2**32 - 1))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def worker_init_fn_from_seed(base_seed: int) -> Callable[[int], None]:
    """DataLoader worker_init_fn：每个 worker 使用不同派生种子。"""

    def _fn(worker_id: int) -> None:
        w = int(base_seed) + int(worker_id) + 1
        random.seed(w)
        np.random.seed(w % (2**32 - 1))
        torch.manual_seed(w)

    return _fn


def dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
