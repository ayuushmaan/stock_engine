"""Unit tests for config/settings.py and environment loading."""
from pathlib import Path
from config.settings import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    DATA_FINAL,
    MODELS_DIR,
    OUTPUTS_DIR,
    OUTPUTS_TABLES,
    OUTPUTS_FIGURES,
    GCP_PROJECT_ID,
    ALPHA,
    DIRECTION_THRESHOLD,
    seed_everything,
)


def test_paths_exist():
    """Verify that all core directories are instantiated properly."""
    assert isinstance(PROJECT_ROOT, Path)
    assert DATA_RAW.exists()
    assert DATA_PROCESSED.exists()
    assert DATA_FINAL.exists()
    assert MODELS_DIR.exists()
    assert OUTPUTS_TABLES.exists()
    assert OUTPUTS_FIGURES.exists()


def test_core_parameters():
    """Verify signal and calibration parameters have sensible bounds."""
    assert ALPHA > 0.0
    assert 0.0 < DIRECTION_THRESHOLD < 1.0
    assert GCP_PROJECT_ID != ""


def test_seed_everything():
    """Verify reproducibility seed setter runs without error."""
    seed_everything(42)
