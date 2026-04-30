from __future__ import annotations

from pathlib import Path

import yaml

from cogen_align.utils.experiment_record import write_experiment_record


def test_write_experiment_record(tmp_path: Path) -> None:
    cfg = {"a": 1, "b": {"c": 2}}
    p = write_experiment_record(
        tmp_path,
        cfg=cfg,
        config_path=Path("/tmp/fake.yaml"),
        argv=["train.py", "--config", "x.yaml"],
        repo_root=tmp_path,
    )
    assert p.is_file()
    meta = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert meta["argv"] == ["train.py", "--config", "x.yaml"]
    assert (tmp_path / "experiment_config.yaml").is_file()
