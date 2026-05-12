from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from GoogleNews import GoogleNews

class SentimentEngine:
    def __init__(self):
        # We use FinBERT: A model trained specifically for financial text
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.googlenews = GoogleNews(lang='en', region='IN', period='1d')

    def get_headlines(self, ticker):
        self.googlenews.clear()
        # "share news" often gives better financial results than "stock news" in India
        self.googlenews.search(f"{ticker} share news")
        results = self.googlenews.result()
        return [res['title'] for res in results] if results else []

    def analyze_sentiment(self, headlines):
        if not headlines: return 0
        
        # Convert text to numbers for the AI
        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert output to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Labels: 0=Positive, 1=Negative, 2=Neutral
        # Score = Average(Positive - Negative)
        mean_probs = probs.mean(dim=0)
        sentiment_score = mean_probs[0].item() - mean_probs[1].item()
        return sentiment_score