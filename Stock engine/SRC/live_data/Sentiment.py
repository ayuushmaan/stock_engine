from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pytz
import torch
from GoogleNews import GoogleNews
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from SRC.common.model_config import load_trained_parameters


class SentimentEngine:
    def __init__(self, use_time_filter=True, alpha=1.8, sponsored_penalty=None):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.googlenews = GoogleNews(lang="en", region="IN", period="1d")
        self.use_time_filter = use_time_filter
        self.ist = pytz.timezone("Asia/Kolkata")
        self.alpha = float(alpha)
        self.direction_threshold = 0.10
        self.model_parameters = load_trained_parameters()
        if sponsored_penalty is None:
            sponsored_penalty = self.model_parameters.sponsored_news_penalty
        self.sponsored_penalty = max(0.0, float(sponsored_penalty))
        self.sponsored_keywords = [
            "sponsored",
            "partner content",
            "paid post",
            "promoted",
            "brand studio",
            "advertorial",
            "advt",
            "paid content",
        ]

    def get_valid_news_window(self):
        now = datetime.now(self.ist)
        if now.hour < 9:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=8)
            end = now.replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            start = now.replace(hour=16, minute=0, second=0, microsecond=0)
            end = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return start, end

    def is_in_valid_window(self):
        now = datetime.now(self.ist)
        return now.hour < 9

    def filter_news_by_time(self, results):
        if not self.use_time_filter or not results:
            return results
        filtered = []
        for res in results:
            try:
                filtered.append(res)
            except Exception:
                continue
        return filtered

    def detect_sponsored(self, article: Dict) -> bool:
        title = (article.get("title") or "").lower()
        source = (article.get("media") or article.get("source") or "").lower()
        snippet = (article.get("desc") or "").lower()
        searchable = f"{title} {source} {snippet}"
        return any(keyword in searchable for keyword in self.sponsored_keywords)

    def get_news_results(self, ticker: str) -> List[Dict]:
        self.googlenews.clear()
        self.googlenews.search(f"{ticker} share news")
        results = self.googlenews.result() or []
        filtered_results = self.filter_news_by_time(results)
        enriched = []
        for res in filtered_results:
            enriched.append(
                {
                    "symbol": ticker,
                    "title": res.get("title", "").strip(),
                    "source": res.get("media", "GoogleNews"),
                    "published_raw": res.get("date", ""),
                    "link": res.get("link", ""),
                    "is_sponsored": int(self.detect_sponsored(res)),
                }
            )
        return [row for row in enriched if row["title"]]

    @staticmethod
    def score_to_direction(score: float, threshold: float = 0.10) -> str:
        if score > threshold:
            return "BULLISH"
        if score < -threshold:
            return "BEARISH"
        return "NEUTRAL"

    def score_to_intensity_label(self, score: float) -> str:
        if score >= 0.60:
            return "HEAVY_GAIN"
        if score >= 0.20:
            return "GAIN"
        if score <= -0.60:
            return "HEAVY_FALL"
        if score <= -0.20:
            return "FALL"
        return "FLAT"

    def normalize_raw_signal(self, raw_signal: float) -> float:
        return float(np.clip(np.tanh(self.alpha * raw_signal), -1.0, 1.0))

    def analyze_articles(self, articles: List[Dict]) -> Dict:
        if not articles:
            return {
                "articles": [],
                "all_sentiment_raw": 0.0,
                "pred_score": 0.0,
                "pred_direction": "NEUTRAL",
                "pred_intensity": "FLAT",
                "sponsored_sentiment_raw": None,
                "non_sponsored_sentiment_raw": None,
                "sponsored_pred_score": None,
                "non_sponsored_pred_score": None,
                "sponsored_count": 0,
                "non_sponsored_count": 0,
            }

        titles = [article["title"] for article in articles]
        inputs = self.tokenizer(titles, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

        for idx, article in enumerate(articles):
            pos_prob = float(probs[idx][0])
            neg_prob = float(probs[idx][1])
            neu_prob = float(probs[idx][2])
            article_score = pos_prob - neg_prob
            adjusted_score = article_score * (1 - self.sponsored_penalty * article["is_sponsored"])
            article["positive_prob"] = pos_prob
            article["negative_prob"] = neg_prob
            article["neutral_prob"] = neu_prob
            article["sentiment_score"] = float(article_score)
            article["adjusted_sentiment_score"] = float(adjusted_score)

        all_scores = [article["adjusted_sentiment_score"] for article in articles]
        all_raw = float(np.mean(all_scores))
        pred_score = self.normalize_raw_signal(all_raw)

        sponsored_scores = [a["adjusted_sentiment_score"] for a in articles if a["is_sponsored"] == 1]
        non_sponsored_scores = [a["adjusted_sentiment_score"] for a in articles if a["is_sponsored"] == 0]

        sponsored_raw = float(np.mean(sponsored_scores)) if sponsored_scores else None
        non_sponsored_raw = float(np.mean(non_sponsored_scores)) if non_sponsored_scores else None

        return {
            "articles": articles,
            "all_sentiment_raw": all_raw,
            "pred_score": pred_score,
            "pred_direction": self.score_to_direction(pred_score, self.direction_threshold),
            "pred_intensity": self.score_to_intensity_label(pred_score),
            "sponsored_sentiment_raw": sponsored_raw,
            "non_sponsored_sentiment_raw": non_sponsored_raw,
            "sponsored_pred_score": self.normalize_raw_signal(sponsored_raw) if sponsored_raw is not None else None,
            "non_sponsored_pred_score": self.normalize_raw_signal(non_sponsored_raw) if non_sponsored_raw is not None else None,
            "sponsored_count": len(sponsored_scores),
            "non_sponsored_count": len(non_sponsored_scores),
            "sponsored_penalty": self.sponsored_penalty,
            "model_source": self.model_parameters.source,
        }
