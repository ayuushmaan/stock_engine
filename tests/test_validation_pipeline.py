"""Tests for LLM labeler and human validation inter-annotator agreement modules."""
import numpy as np
import pandas as pd
import pytest

from pipeline.llm_labeler import mock_llm_judge
from pipeline.human_validation import compute_gwet_ac1, evaluate_agreement


def test_mock_llm_judge_pr_wire():
    """Verify mock judge correctly classifies PR wires as SPONSORED."""
    res = mock_llm_judge(
        headline="Quarterly business summary",
        text="All forward-looking statements.",
        domain="businesswire.com",
        url="https://businesswire.com/press-release",
    )
    assert res["label"] == "SPONSORED"
    assert res["is_sponsored"] is True
    assert res["confidence"] >= 0.70


def test_mock_llm_judge_organic():
    """Verify mock judge classifies regular news as ORGANIC."""
    res = mock_llm_judge(
        headline="Tech stocks slump as inflation fears mount",
        text="The tech sector led losses across Asian indices today after central bank commentary.",
        domain="bloomberg.com",
        url="https://bloomberg.com/news/tech-slump",
    )
    assert res["label"] == "ORGANIC"
    assert res["is_sponsored"] is False


def test_gwet_ac1_perfect_agreement():
    """Verify Gwet's AC1 equals 1.0 for perfect agreement."""
    y1 = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    y2 = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    ac1 = compute_gwet_ac1(y1, y2)
    assert pytest.approx(ac1, 1e-5) == 1.0


def test_evaluate_agreement_dataframe():
    """Verify full evaluation pipeline returns valid metric dictionary."""
    df = pd.DataFrame({
        "hand_label": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "llm_label":  [1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
        "weak_label": [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    })
    res = evaluate_agreement(df)
    assert res["n_samples"] == 10
    assert "human_vs_llm" in res["comparisons"]
    assert "cohen_kappa" in res["comparisons"]["human_vs_llm"]
    assert "gwet_ac1" in res["comparisons"]["human_vs_llm"]
