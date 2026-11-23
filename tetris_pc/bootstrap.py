"""Pozwala uruchamiać skrypty zwykłym ``python``: przełącza na projektowy .venv."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def use_project_venv(script: str) -> None:
    if getattr(sys, "frozen", False):
        return
    root =Path(__file__).resolve().parent.parent / ".venv"
    python = root / "Scripts" / "python.exe"
    if python.exists() and Path(sys.prefix).resolve() != root.resolve():
        raise SystemExit(subprocess.call([str(python), str(Path(script).resolve()), *sys.argv[1:]]))
