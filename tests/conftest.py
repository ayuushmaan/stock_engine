"""Pytest configuration and shared fixtures for Stock Engine test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_articles_df() -> pd.DataFrame:
    """Fixture providing synthetic article dataset for unit testing."""
    return pd.DataFrame({
        "ticker": ["RELIANCE", "RELIANCE", "RELIANCE", "INFY", "INFY"],
        "effective_date": ["2024-01-05", "2024-01-05", "2024-01-05", "2024-01-05", "2024-01-05"],
        "tone_score": [5.0, -2.0, 3.5, -6.0, -4.0],
        "sponsored_prob": [0.1, 0.8, 0.2, 0.15, 0.9],
        "time_bucket": ["CLOSED_POST", "OPEN", "CLOSED_PRE", "OPEN", "CLOSED_POST"],
        "domain": ["economictimes.indiatimes.com", "businesswire.com", "livemint.com", "reuters.com", "prnewswire.com"],
        "url": [
            "https://economictimes.indiatimes.com/news/reliance-q3",
            "https://businesswire.com/news/reliance-brandstudio-release",
            "https://livemint.com/market/reliance-petro-expansion",
            "https://reuters.com/business/infy-guidance-cut",
            "https://prnewswire.com/infy-csr-milestone",
        ],
    })


@pytest.fixture
def sample_single_stock_articles() -> pd.DataFrame:
    """Fixture providing 4 articles for a single stock."""
    return pd.DataFrame({
        "ticker": ["TCS", "TCS", "TCS", "TCS"],
        "effective_date": ["2024-02-01"] * 4,
        "tone_score": [4.0, 6.0, 2.0, -1.0],
        "sponsored_prob": [0.05, 0.15, 0.20, 0.10],
        "time_bucket": ["CLOSED_PRE", "CLOSED_POST", "OPEN", "OPEN"],
    })
