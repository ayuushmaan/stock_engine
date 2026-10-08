"""Unit tests for pipeline/scoring_engine.py and morphological extractors."""
import pytest
from pipeline.scoring_engine import (
    score_single_article,
    DEFAULT_WEIGHTS,
    DEFAULT_BIAS,
)
from pipeline.extractors.news_morphometrics import (
    extract_all_morphometrics,
    extract_cta_density,
    extract_promo_density,
)


def test_promo_density_extraction():
    """Test promotional keyword density calculation."""
    text = "We are proud to announce record milestone and best-ever growth in our history."
    density = extract_promo_density(text)
    assert density > 0.0

    neutral_text = "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent on Thursday."
    neutral_density = extract_promo_density(neutral_text)
    assert neutral_density == 0.0


def test_cta_density_extraction():
    """Test Call-To-Action density extractor."""
    promo_cta = "Click here to buy now and subscribe for exclusive offers."
    assert extract_cta_density(promo_cta) > 0.0

    news_text = "TCS reported quarterly net profit of 12000 crores."
    assert extract_cta_density(news_text) == 0.0


def test_pr_wire_invariant_override():
    """Test Layer 0 override: PR wire domain forces S(A) to 1.0."""
    res = score_single_article(
        headline="Company signs new contract",
        text="Normal business update.",
        domain="businesswire.com",
        url="https://businesswire.com/news/123",
    )
    assert res["score"] == 1.0
    assert res["classification"] == "sponsored"
    assert "Layer 0" in res["layer_triggered"]


def test_organic_news_scoring():
    """Test pure financial news gets low sponsored score."""
    res = score_single_article(
        headline="RBI keeps interest rates unchanged amid inflation data",
        text="The monetary policy committee voted 5-1 to maintain the benchmark repo rate at 6.5%.",
        domain="reuters.com",
        url="https://reuters.com/markets/rbi-rates",
    )
    assert res["score"] < 0.35
    assert res["classification"] in ["organic", "unclear"]
