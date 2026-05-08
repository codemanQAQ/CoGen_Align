"""将 stdout/stderr 同步写入文件，便于 tmux/无滚动条环境查看完整报错。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO

_attached: IO[str] | None = None


class _Tee(IO[str]):
    __slots__ = ("_term", "_file")

    def __init__(self, term: IO[str], log_fp: IO[str]) -> None:
        self._term = term
        self._file = log_fp

    def write(self, s: str) -> int:
        self._term.write(s)
        self._file.write(s)
        self._term.flush()
        self._file.flush()
        return len(s)

    def flush(self) -> None:
        self._term.flush()
        self._file.flush()

    def fileno(self) -> int:
        return self._term.fileno()

    def isatty(self) -> bool:
        return self._term.isatty()

    def writable(self) -> bool:
        return True


def attach_console_log_file(path: str | Path) -> Path:
    """
    rank0 在训练开始处调用一次：之后 print / traceback 会同时进终端与 path。
    未捕获异常的 traceback 走 sys.stderr，同样会写入该文件。
    多进程时请勿在非 0 rank 调用（否则会争用同一文件）。
    """
    global _attached
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _attached is not None:
        return p
    banner = (
        f"\n{'=' * 72}\n"
        f"console log: {p.resolve()}\n"
        f"{'=' * 72}\n"
    )
    log_fp = open(p, "a", encoding="utf-8", errors="replace", buffering=1)
    log_fp.write(banner)
    log_fp.flush()
    term_out = sys.__stdout__
    term_err = sys.__stderr__
    sys.stdout = _Tee(term_out, log_fp)  # type: ignore[assignment]
    sys.stderr = _Tee(term_err, log_fp)  # type: ignore[assignment]
    _attached = log_fp
    return p


def detach_console_log() -> None:
    """一般无需调用；便于单测恢复 sys.stdout。"""
    global _attached
    if _attached is None:
        return
    try:
        sys.stdout = sys.__stdout__  # type: ignore[assignment]
        sys.stderr = sys.__stderr__  # type: ignore[assignment]
        _attached.close()
    finally:
        _attached = None
