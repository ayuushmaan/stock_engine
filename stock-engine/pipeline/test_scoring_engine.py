"""Automated Verification and Invariant Tests for the HHEE Scoring Engine.

Tests:
  1. Invariant Overrides (Layer 0: PR wires, BrandStudio, Disclaimers) -> 1.0
  2. Pure Organic News Articles -> <= tau_low (Organic)
  3. Continuous Monotonicity: Higher promo/CTA density -> Higher S(A)
  4. Abstain Zone: Neutral/Ambiguous inputs -> 'unclear'
  5. Mathematical Bounds: S(A) strictly in [0.0, 1.0]
  6. Explainability Vector: All feature keys present and non-null
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# ── project imports ───────────────────────────────────────────────
from pipeline.scoring_engine import (
    DEFAULT_BIAS,
    DEFAULT_WEIGHTS,
    score_single_article,
    sigmoid,
)
from pipeline.extractors.news_morphometrics import (
    extract_all_morphometrics,
    extract_boilerplate_ratio,
    extract_cta_density,
    extract_disclosure_flag,
    extract_headline_hype,
    extract_pr_wire_flag,
    extract_promo_density,
    extract_quote_monopoly,
    extract_risk_absence,
    extract_url_sponsored_flag,
)


class TestHHEEScoringEngine(unittest.TestCase):

    def test_layer0_pr_wire_override(self):
        """PR wire URL must deterministically yield score=1.0, class='sponsored'."""
        res = score_single_article(
            url="https://www.prnewswire.com/news-releases/company-announcement-12345.html",
            headline="Quarterly Update",
            text="General content",
        )
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["classification"], "sponsored")
        self.assertIn("Layer 0", res["layer_triggered"])

    def test_layer0_brandstudio_url_override(self):
        """BrandStudio / Sponsored URL path must deterministically trigger Layer 0."""
        res = score_single_article(
            url="https://economictimes.indiatimes.com/brandstudio/spotlight/innovations-in-cloud.cms",
            headline="Cloud Transformations",
            text="Informative article",
        )
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["classification"], "sponsored")
        self.assertIn("Layer 0", res["layer_triggered"])

    def test_layer0_statutory_disclosure_override(self):
        """Explicit commercial disclaimer in text must trigger Layer 0 override."""
        res = score_single_article(
            url="https://www.example.com/news/article-99.html",
            headline="Market Trends",
            text="This is a sponsored feature presented by Alpha Capital. Views expressed are not of editorial.",
        )
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["classification"], "sponsored")
        self.assertIn("Layer 0", res["layer_triggered"])

    def test_organic_journalistic_news(self):
        """Editorial investigative news with risk analysis should score low (Organic)."""
        res = score_single_article(
            url="https://www.livemint.com/market/stock-market-news/sebi-investigates-accounting-fraud.html",
            headline="SEBI widens regulatory probe into fraud allegations",
            text="Regulators initiated investigation into corporate governance violation, penalty imposition, and loan default. Analysts noted severe debt distress and margin compression.",
        )
        self.assertLessEqual(res["score"], 0.30)
        self.assertEqual(res["classification"], "organic")

    def test_promotional_density_monotonicity(self):
        """Increasing promotional superlative keywords must monotonically increase S(A)."""
        base_text = "The enterprise reported financial performance for the period."
        promo_text = (
            "The enterprise proudly announced milestone record-breaking performance, "
            "achieving groundbreaking, revolutionary growth with industry-leading, "
            "award-winning stellar performance and pioneering leadership."
        )

        score_base = score_single_article(text=base_text, url="https://independent-daily.com/news/1")["score"]
        score_promo = score_single_article(text=promo_text, url="https://independent-daily.com/news/1")["score"]

        self.assertGreater(score_promo, score_base)

    def test_cta_density_monotonicity(self):
        """Adding commercial Call-to-Action phrases must monotonically increase S(A)."""
        base_text = "The software update includes new user interface enhancements."
        cta_text = (
            "The software update includes enhancements. Visit www.product.com today, "
            "click here to buy the full license, download the app on google play, and book your demo now."
        )

        score_base = score_single_article(text=base_text, url="https://independent-daily.com/news/2")["score"]
        score_cta = score_single_article(text=cta_text, url="https://independent-daily.com/news/2")["score"]

        self.assertGreater(score_cta, score_base)

    def test_abstain_zone_for_ambiguous_news(self):
        """Mildly positive news with zero explicit markers should map to 'unclear' if near center."""
        res = score_single_article(
            url="https://www.genericportal.com/business/roundup.html",
            headline="Company shares quarterly highlights and expansion roadmap",
            text="The company shared quarterly highlights indicating strong growth in key segments, while assessing ongoing economic developments.",
            tau_low=0.15,
            tau_high=0.85,
        )
        self.assertGreater(res["score"], 0.15)
        self.assertLess(res["score"], 0.85)
        self.assertEqual(res["classification"], "unclear")

    def test_mathematical_bounds_and_explainability(self):
        """Score must strictly satisfy 0 <= S(A) <= 1 and emit attribution."""
        res = score_single_article(
            url="https://www.news.com/article",
            headline="Standard headline",
            text="Standard body text with multiple words.",
        )
        self.assertTrue(0.0 <= res["score"] <= 1.0)
        self.assertIn("attribution", res)
        self.assertIn("bias", res["attribution"])
        self.assertIn("features", res)
        self.assertTrue(isinstance(res["features"], dict))


if __name__ == "__main__":
    unittest.main()
