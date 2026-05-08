from __future__ import annotations

from cogen_align.utils.console_log import attach_console_log_file, detach_console_log


def test_attach_console_log_file_tee(tmp_path):
    log = tmp_path / "c.log"
    attach_console_log_file(log)
    print("hello-tee", flush=True)
    detach_console_log()
    text = log.read_text(encoding="utf-8")
    assert "hello-tee" in text
    assert "console log:" in text
