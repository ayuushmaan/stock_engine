"""Pipeline Step 11 — Scrape article bodies for gold + LLM sets (700 URLs max).

Why this exists:
  GDELT GKG gives URL + tone, NOT body/title. HHEE morphometrics
  (boilerplate, quote_monopoly, cta_density) currently fall back to
  scoring the URL string. This script fetches real bodies for the
  200 hand labels + 500 LLM annotations so weights can be LEARNED.

Inputs:
  data/final/hand_labels.parquet (source_url)
  data/final/llm_annotations.parquet (url)

Output:
  data/final/article_bodies.parquet (url, title, body, word_count_scraped, fetch_status, scraped_at)

Design (hiring-safe):
  - SHA-256 disk cache in data/processed/scrape_cache/ (no re-fetch)
  - Polite: 1s delay, real User-Agent, 15s timeout, skip on 403/paywall
  - Fail-open: never crash run; record fetch_status per URL
  - Uses requests + bs4 (already in venv). No new deps.

Usage:
  python pipeline/11_scrape_bodies.py
  python pipeline/11_scrape_bodies.py --limit 20   # smoke test
  python pipeline/11_scrape_bodies.py --no-cache   # force refetch
"""
from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import DATA_FINAL, DATA_PROCESSED, setup_logging

logger = setup_logging()
CACHE_DIR = DATA_PROCESSED / "scrape_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockEngineResearch/0.1 (academic NLP audit; contact: research@local)",
    "Accept": "text/html,application/xhtml+xml",
}


def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{h}.html"


def fetch_url(url: str, use_cache: bool = True) -> tuple[str | None, str]:
    """Fetch raw HTML. Returns (html_or_None, status)."""
    cp = _cache_path(url)
    if use_cache and cp.exists() and cp.stat().st_size > 500:
        try:
            return cp.read_text(encoding="utf-8", errors="ignore"), "cache_hit"
        except Exception:
            pass
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        html = r.text
        if len(html) < 500:
            return None, "empty_body"
        try:
            cp.write_text(html, encoding="utf-8")
        except Exception:
            pass
        return html, "fetched"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        return None, f"req_err:{type(e).__name__}"


def extract_title_body(html: str) -> tuple[str, str]:
    """Minimal boilerplate removal: title + <p> text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    # prefer article/main, fallback to all <p>
    container = soup.find("article") or soup.find("main") or soup
    paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    paras = [p for p in paras if len(p.split()) >= 8]  # drop nav cruft
    body = "\n".join(paras)[:20000]  # cap at ~20k chars
    return title, body


def collect_urls() -> pd.DataFrame:
    urls: set[str] = set()
    hp = DATA_FINAL / "hand_labels.parquet"
    lp = DATA_FINAL / "llm_annotations.parquet"
    if hp.exists():
        h = pd.read_parquet(hp)
        col = "source_url" if "source_url" in h.columns else "url"
        urls.update(h[col].dropna().astype(str).str.strip().tolist())
    if lp.exists():
        l = pd.read_parquet(lp)
        col = "url" if "url" in l.columns else "source_url"
        urls.update(l[col].dropna().astype(str).str.strip().tolist())
    urls = {u for u in urls if u.startswith("http") and len(u) > 12}
    return pd.DataFrame({"url": sorted(urls)})


def run(limit: int | None = None, use_cache: bool = True, delay: float = 1.0) -> None:
    df = collect_urls()
    logger.info(f"Collected {len(df)} unique URLs")
    if limit:
        df = df.head(limit)
        logger.info(f"Smoke-test limit: {len(df)} URLs")
    from tqdm import tqdm

    rows = []
    for url in tqdm(df["url"], desc="Scraping"):
        html, status = fetch_url(url, use_cache=use_cache)
        title, body = ("", "")
        if html:
            try:
                title, body = extract_title_body(html)
            except Exception as e:
                status = f"parse_err:{type(e).__name__}"
        rows.append({
            "url": url,
            "title": title,
            "body": body,
            "word_count_scraped": len(body.split()) if body else 0,
            "fetch_status": status if not body else (status + "+parsed" if "+" not in status else status),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        })
        if status == "fetched":
            time.sleep(delay)
    out = pd.DataFrame(rows)
    ok = (out["word_count_scraped"] > 50).sum()
    logger.info(f"Parsed bodies >50 words: {ok}/{len(out)} ({100*ok/max(1,len(out)):.1f}%)")
    logger.info(out["fetch_status"].value_counts().head(10).to_string())
    out_path = DATA_FINAL / "article_bodies.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(out)} rows)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape gold-set article bodies.")
    p.add_argument("--limit", type=int, default=None, help="Smoke-test first N URLs")
    p.add_argument("--no-cache", action="store_true", help="Force refetch ignoring cache")
    p.add_argument("--delay", type=float, default=1.0, help="Delay secs between fresh fetches")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(limit=a.limit, use_cache=not a.no_cache, delay=a.delay)
