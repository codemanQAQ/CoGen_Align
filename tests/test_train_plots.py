"""train_plots：无头 PNG 输出。"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from cogen_align.utils.train_plots import save_stage1_curves, save_stage2_curves


def test_save_stage1_curves(tmp_path):
    p = save_stage1_curves(
        tmp_path,
        step=99,
        train_rows=[
            {"step": 0, "loss": 2.0, "lr": 1e-4, "grad_norm": 1.0, "top1_acc": 0.1},
            {"step": 50, "loss": 1.5, "lr": 1e-4, "grad_norm": 0.9, "top1_acc": 0.2},
        ],
        val_rows=[{"step": 50, "top1_a2t": 0.05, "top1_t2a": 0.04}],
    )
    assert p.is_file()
    assert (tmp_path / "plots" / "curves_latest.png").is_file()


def test_save_stage2_curves(tmp_path):
    p = save_stage2_curves(
        tmp_path,
        step=10,
        train_rows=[
            {"step": 0, "loss": 3.0, "lr": 1e-5, "grad_norm": 0.5},
        ],
        val_rows=[{"split": "val", "step": 0, "wer": 0.8}],
    )
    assert p.is_file()
