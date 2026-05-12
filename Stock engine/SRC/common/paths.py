from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    # SRC is located at <root>/SRC
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
OUTPUTS_DIR = ROOT / "outputs"
LATEST_DIR = OUTPUTS_DIR / "latest"
HISTORY_DIR = OUTPUTS_DIR / "history"
RESEARCH_DIR = OUTPUTS_DIR / "research"
PAPER_DIR = OUTPUTS_DIR / "paper"
DB_DIR = ROOT / "db"
SCRIPTS_DIR = ROOT / "scripts"
LOGS_DIR = ROOT / "logs"


def ensure_dirs() -> None:
    for path in [OUTPUTS_DIR, LATEST_DIR, HISTORY_DIR, RESEARCH_DIR, PAPER_DIR, DB_DIR, SCRIPTS_DIR, LOGS_DIR]:
        os.makedirs(path, exist_ok=True)
