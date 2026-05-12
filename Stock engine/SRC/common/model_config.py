from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Optional

from SRC.common.paths import ROOT


@dataclass(frozen=True)
class ModelParameters:
    sponsored_news_penalty: float = 0.0
    calibration_k: float = 0.0
    improvement_over_baseline: Optional[float] = None
    source: str = "defaults"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_trained_parameters(config_path: Optional[Path] = None) -> ModelParameters:
    """Load historical-training parameters for the live pipeline if available."""
    target_path = config_path or (ROOT / "ml_trained_config.py")
    if not target_path.exists():
        return ModelParameters()

    spec = util.spec_from_file_location("ml_trained_config", target_path)
    if spec is None or spec.loader is None:
        return ModelParameters()

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    improvement = getattr(module, "IMPROVEMENT_OVER_BASELINE", None)
    return ModelParameters(
        sponsored_news_penalty=_safe_float(getattr(module, "SPONSORED_NEWS_PENALTY", 0.0)),
        calibration_k=_safe_float(getattr(module, "CALIBRATION_K", 0.0)),
        improvement_over_baseline=_safe_float(improvement, None) if improvement is not None else None,
        source=str(target_path),
    )
