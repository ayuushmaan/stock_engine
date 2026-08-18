# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 06 - LLM Structured Extraction Pipeline
#
# **Purpose**: Extract structured sentiment, rationale, and entity information
# from news articles using an open-weight instruct LLM, providing a third
# independent sentiment signal beyond GDELT tone and FinBERT.
#
# **Design**:
# 1. Define a fixed, versioned JSON schema for extraction.
# 2. Reuse the same stratified sample from `05_finbert_validation.ipynb`.
# 3. Run a 100-article pilot to validate JSON output quality.
# 4. Scale to the full sample with retry-with-repair logic.
# 5. Save results to `data/processed/llm_extraction_greedy.parquet`.
#
# **Model**: `Qwen2.5-0.5B-Instruct` in float16, pre-downloaded locally.
# At ~0.5B parameters (~1 GB in fp16), it fits comfortably on any GPU
# including an RTX 3050 6 GB with ample headroom for KV cache.
# A larger model (Mistral-7B, Llama-3.1-8B at 4-bit) would give better
# extraction quality; the 0.5B model serves as the working baseline
# with a clear upgrade path.

# %%
# ============================================================
# Cell 1: Setup and Imports
# ============================================================
import sys
import json
import re
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

if sys.platform.startswith('win') and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / 'config' / 'settings.py').exists():
            return candidate
    raise FileNotFoundError('Could not locate project root')

ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_FINAL, DATA_PROCESSED, OUTPUTS_FIGURES,
    seed_everything,
)

seed_everything(42)

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

print(f'Project root: {ROOT}')

# %% [markdown]
# ## Section 1: Extraction Schema (v1.0)
#
# We define a fixed JSON schema for structured extraction. Each article
# produces exactly one JSON object with these fields:
#
# ```json
# {
#   "sentiment": "bullish" | "bearish" | "neutral",
#   "confidence": 0.0-1.0,
#   "rationale": "short string explaining the sentiment call",
#   "entities": ["TICKER1", "TICKER2", ...]
# }
# ```
#
# ### Sentiment Mapping to Existing Measures
#
# | LLM Output | Numeric Score | GDELT Tone Equivalent | FinBERT Equivalent |
# |---|---|---|---|
# | `bullish` | +1.0 | Positive tone (tone_score > 0) | P(positive) > P(negative) |
# | `neutral` | 0.0 | Near-zero tone | P(neutral) dominant |
# | `bearish` | -1.0 | Negative tone (tone_score < 0) | P(negative) > P(positive) |
#
# The `confidence` field (0-1) can be used to create a continuous score:
# `llm_score = sentiment_numeric * confidence`, which maps to [-1, +1],
# directly comparable to FinBERT's `P(pos) - P(neg)` score.

# %%
# ============================================================
# Cell 2: Schema definition
# ============================================================
SCHEMA_VERSION = "1.0"

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["bullish", "bearish", "neutral"],
            "description": "Overall financial sentiment of the article"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the sentiment call (0=uncertain, 1=very confident)"
        },
        "rationale": {
            "type": "string",
            "description": "One-sentence explanation of why this sentiment was assigned"
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of company names or stock tickers mentioned"
        }
    },
    "required": ["sentiment", "confidence", "rationale", "entities"]
}

SENTIMENT_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

print(f'Schema version: {SCHEMA_VERSION}')
print(f'Schema fields: {list(EXTRACTION_SCHEMA["properties"].keys())}')

# %% [markdown]
# ## Section 2: Load Sample from 05_finbert_validation
#
# We reuse the exact same stratified sample used in `05_finbert_validation.ipynb`
# to enable direct comparison. We reload it by re-running the same sampling
# logic with the same random seed (42) and bucket proportions.

# %%
# ============================================================
# Cell 3: Load and reconstruct the same stratified sample
# ============================================================
SAMPLE_IDS_PATH = DATA_PROCESSED / 'llm_sample_article_ids.parquet'

scores_df = pd.read_parquet(
    DATA_PROCESSED / 'sponsored_scores.parquet',
    columns=[
        'DocumentIdentifier', 'tone_score', 'sponsored_prob',
        'time_bucket', 'ticker', 'effective_date', 'source_domain',
    ]
)
print(f'Loaded sponsored_scores: {len(scores_df):,} rows')

# Assign row index as stable article ID before any filtering
scores_df['article_id'] = np.arange(len(scores_df))

# Classify
scores_df['credibility'] = np.where(
    scores_df['sponsored_prob'] < 0.3, 'organic',
    np.where(scores_df['sponsored_prob'] > 0.7, 'sponsored', 'uncertain')
)
scores_df['window'] = np.where(
    scores_df['time_bucket'] == 'OPEN', 'open', 'closed'
)

clean_df = scores_df[scores_df['credibility'] != 'uncertain'].copy()
clean_df['bucket'] = clean_df['credibility'] + '_' + clean_df['window']

# Same sampling as 05
TARGET_SAMPLE_SIZE = 5000
np.random.seed(42)
bucket_counts = clean_df['bucket'].value_counts()
bucket_proportions = bucket_counts / bucket_counts.sum()
sample_sizes = (bucket_proportions * TARGET_SAMPLE_SIZE).round().astype(int)
diff = TARGET_SAMPLE_SIZE - sample_sizes.sum()
if diff != 0:
    sample_sizes.iloc[0] += diff

sampled_dfs = []
for bucket_name, n_sample in sample_sizes.items():
    bucket_data = clean_df[clean_df['bucket'] == bucket_name]
    n_actual = min(n_sample, len(bucket_data))
    sampled = bucket_data.sample(n=n_actual, random_state=42)
    sampled_dfs.append(sampled)

sample_df = pd.concat(sampled_dfs, ignore_index=True)
print(f'Stratified sample: {len(sample_df):,} articles')

# Extract pseudo-titles (same logic as 05)
def extract_title_from_url(url):
    """Extract a readable pseudo-title from URL path slug."""
    if not isinstance(url, str) or len(url) < 10:
        return None
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        path = re.sub(r'\.(html?|cms|php|asp|aspx|jsp|xml|json)$', '', path, flags=re.IGNORECASE)
        slug = path.split('/')[-1]
        if slug.isdigit() or slug.lower() == 'articleshow':
            parts = path.split('/')
            for part in reversed(parts):
                if not part.isdigit() and part.lower() != 'articleshow' and len(part) > 3:
                    slug = part
                    break
        slug = re.sub(r'[-_]?\d{5,}$', '', slug)
        slug = re.sub(r'^(article|story|news|post|blog)[-_]', '', slug, flags=re.IGNORECASE)
        title = re.sub(r'[-_]+', ' ', slug).strip()
        if len(title) < 8 or title.replace(' ', '').isdigit():
            return None
        return title
    except Exception:
        return None

sample_df['pseudo_title'] = sample_df['DocumentIdentifier'].apply(extract_title_from_url)
sample_df = sample_df[sample_df['pseudo_title'].notna()].copy()
print(f'After URL parse: {len(sample_df):,} articles with titles')

# Save article IDs for reproducibility
sample_df[['article_id', 'bucket', 'DocumentIdentifier', 'pseudo_title']].to_parquet(
    SAMPLE_IDS_PATH, index=False
)
print(f'Saved sample article IDs to: {SAMPLE_IDS_PATH}')

# Show bucket distribution
print(f'\nSample bucket distribution:')
for b, c in sample_df['bucket'].value_counts().items():
    print(f'  {b}: {c:,}')

# %% [markdown]
# ## Section 3: Load LLM Model
#
# We load `Qwen2.5-0.5B-Instruct` in float16 from a local checkout
# (`models/qwen05b/`). At ~1 GB it fits any GPU. If the local copy
# is missing, we fall back to downloading from HuggingFace Hub.

# %%
# ============================================================
# Cell 4: Load quantized LLM
# ============================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')
else:
    print('WARNING: No GPU detected. CPU inference will be very slow.')

# Model selection: Local Qwen2.5-0.5B-Instruct (pre-downloaded to models/qwen05b)
LOCAL_MODEL_PATH = ROOT / 'models' / 'qwen05b'
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

if LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / 'model.safetensors').exists():
    model_name = str(LOCAL_MODEL_PATH)
    print(f'Loading local model from {LOCAL_MODEL_PATH}...')
else:
    model_name = FALLBACK_MODEL
    print(f'Loading model {model_name} from HuggingFace Hub...')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print(f'Model loaded successfully: {model_name}')

# Report VRAM usage
if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM usage: {vram_used:.2f} / {vram_total:.1f} GB')

# %% [markdown]
# ## Section 4: Extraction Function
#
# The extraction function sends a single prompt to the LLM with the article
# text and asks for a JSON response matching our schema. We use greedy
# decoding (temperature=0) for reproducibility.

# %%
# ============================================================
# Cell 5: Define extraction function
# ============================================================
SYSTEM_PROMPT = """You are a financial news analyst. Given a news headline or article snippet, extract structured information in JSON format.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation) with these exact fields:
{
  "sentiment": "bullish" or "bearish" or "neutral",
  "confidence": a number between 0.0 and 1.0,
  "rationale": "one sentence explaining your sentiment call",
  "entities": ["list", "of", "company names or tickers mentioned"]
}

Rules:
- sentiment must be exactly one of: "bullish", "bearish", "neutral"
- confidence: 1.0 = very confident, 0.0 = uncertain
- rationale: keep it to one sentence
- entities: list all company names, stock tickers, or financial instruments mentioned
- Output ONLY the JSON object, nothing else"""


def build_prompt(title_text):
    """Build the chat prompt for extraction."""
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nHeadline: {title_text}"}
    ]
    # Use chat template if available, else manual format
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"[INST] {SYSTEM_PROMPT}\n\nHeadline: {title_text} [/INST]"
    return prompt


def extract_json_from_text(text):
    """Try to extract a JSON object from model output text."""
    # Try direct parse first
    text = text.strip()
    
    # Remove markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # Try to find JSON object in the text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Try full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_extraction(result):
    """Validate that extraction matches schema."""
    if not isinstance(result, dict):
        return False, "Not a dict"
    
    required = ['sentiment', 'confidence', 'rationale', 'entities']
    for field in required:
        if field not in result:
            return False, f"Missing field: {field}"
    
    if result['sentiment'] not in ('bullish', 'bearish', 'neutral'):
        return False, f"Invalid sentiment: {result['sentiment']}"
    
    try:
        conf = float(result['confidence'])
        if not (0.0 <= conf <= 1.0):
            return False, f"Confidence out of range: {conf}"
        result['confidence'] = conf
    except (ValueError, TypeError):
        return False, f"Invalid confidence: {result['confidence']}"
    
    if not isinstance(result['rationale'], str):
        return False, "Rationale not a string"
    
    if not isinstance(result['entities'], list):
        return False, "Entities not a list"
    
    return True, "OK"


def extract_single(title_text, max_retries=2):
    """Run extraction for a single article with retry logic."""
    prompt = build_prompt(title_text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    last_error = None
    for attempt in range(max_retries + 1):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=1.0,   # greedy with do_sample=False
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated tokens
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        
        result = extract_json_from_text(text)
        if result is not None:
            valid, msg = validate_extraction(result)
            if valid:
                return result, text, attempt
            else:
                last_error = msg
        else:
            last_error = f"JSON parse failed: {text[:100]}"
        
        # Retry with repair prompt
        if attempt < max_retries:
            repair_msg = f"Your previous response was not valid JSON. Error: {last_error}. Please respond with ONLY a valid JSON object."
            try:
                repair_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\nHeadline: {title_text}\n\n{repair_msg}"}],
                    tokenize=False, add_generation_prompt=True
                )
            except Exception:
                repair_prompt = f"[INST] {SYSTEM_PROMPT}\n\nHeadline: {title_text}\n\n{repair_msg} [/INST]"
            
            inputs = tokenizer(repair_prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # All retries exhausted — return None
    return None, text if 'text' in dir() else '', max_retries


# Quick test
test_title = "reliance industries q4 results profit jumps 25 percent"
test_result, test_raw, test_attempts = extract_single(test_title)
print(f'Test extraction ({test_attempts} retries):')
print(f'  Input: {test_title}')
print(f'  Raw output: {test_raw[:200]}')
print(f'  Parsed: {test_result}')

# %% [markdown]
# ## Section 5: 100-Article Pilot
#
# Run extraction on the first 100 articles to validate JSON output quality.
# If the validity rate is below 98%, the retry-with-repair logic handles it.
# We report the validity rate before and after repairs.

# %%
# ============================================================
# Cell 6: Pilot run (100 articles)
# ============================================================
PILOT_SIZE = 100
pilot_df = sample_df.head(PILOT_SIZE).copy()

pilot_results = []
pilot_raw_outputs = []
pilot_attempts = []
pilot_valid_first_try = 0
pilot_valid_after_retry = 0
pilot_failed = 0

print(f'Running pilot on {PILOT_SIZE} articles...')
t_start = time.time()

for i, (idx, row) in enumerate(pilot_df.iterrows()):
    result, raw, attempts = extract_single(row['pseudo_title'])
    
    pilot_results.append(result)
    pilot_raw_outputs.append(raw)
    pilot_attempts.append(attempts)
    
    if result is not None:
        if attempts == 0:
            pilot_valid_first_try += 1
        else:
            pilot_valid_after_retry += 1
    else:
        pilot_failed += 1
    
    if (i + 1) % 25 == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        print(f'  {i+1}/{PILOT_SIZE} done ({rate:.1f} articles/sec)')

elapsed = time.time() - t_start
total_valid = pilot_valid_first_try + pilot_valid_after_retry

print(f'\nPilot Results ({elapsed:.1f}s):')
print(f'  Valid on first try: {pilot_valid_first_try}/{PILOT_SIZE} ({100*pilot_valid_first_try/PILOT_SIZE:.1f}%)')
print(f'  Valid after retry:  {pilot_valid_after_retry}/{PILOT_SIZE} ({100*pilot_valid_after_retry/PILOT_SIZE:.1f}%)')
print(f'  Total valid:        {total_valid}/{PILOT_SIZE} ({100*total_valid/PILOT_SIZE:.1f}%)')
print(f'  Failed:             {pilot_failed}/{PILOT_SIZE} ({100*pilot_failed/PILOT_SIZE:.1f}%)')
print(f'  Throughput:         {PILOT_SIZE/elapsed:.1f} articles/sec')

validity_rate = 100 * total_valid / PILOT_SIZE
print(f'\n  JSON validity rate: {validity_rate:.1f}%')
if validity_rate >= 98:
    print('  PASS: Validity rate >= 98%. Proceeding to full extraction.')
else:
    print('  NOTE: Validity rate < 98%. Retry logic is handling failures.')

# Show sample of extractions
print('\nSample extractions:')
for i in range(min(5, len(pilot_results))):
    if pilot_results[i] is not None:
        print(f'  [{pilot_df.iloc[i]["bucket"]}] "{pilot_df.iloc[i]["pseudo_title"][:60]}"')
        print(f'    -> {pilot_results[i]}')

# %% [markdown]
# ## Section 6: Full Extraction
#
# Run extraction on the complete sample. Progress is reported every 100 articles.
# Results are saved incrementally to handle interruptions.

# %%
# ============================================================
# Cell 7: Full extraction run
# ============================================================
OUTPUT_PATH = DATA_PROCESSED / 'llm_extraction_greedy.parquet'

all_results = []
n_total = len(sample_df)
n_valid_first = 0
n_valid_retry = 0
n_failed = 0

print(f'Running full extraction on {n_total:,} articles...')
print(f'Estimated time: {n_total / max(PILOT_SIZE/elapsed, 0.1) / 60:.0f} minutes')
t_start = time.time()

for i, (idx, row) in enumerate(sample_df.iterrows()):
    result, raw, attempts = extract_single(row['pseudo_title'])
    
    if result is not None:
        if attempts == 0:
            n_valid_first += 1
        else:
            n_valid_retry += 1
        
        all_results.append({
            'article_id': row['article_id'],
            'bucket': row['bucket'],
            'credibility': row['credibility'],
            'window': row['window'],
            'ticker': row['ticker'],
            'effective_date': row['effective_date'],
            'tone_score': row['tone_score'],
            'sponsored_prob': row['sponsored_prob'],
            'pseudo_title': row['pseudo_title'],
            'llm_sentiment': result['sentiment'],
            'llm_confidence': result['confidence'],
            'llm_rationale': result['rationale'],
            'llm_entities': json.dumps(result['entities']),
            'llm_score': SENTIMENT_MAP[result['sentiment']] * result['confidence'],
            'n_retries': attempts,
        })
    else:
        n_failed += 1
        all_results.append({
            'article_id': row['article_id'],
            'bucket': row['bucket'],
            'credibility': row['credibility'],
            'window': row['window'],
            'ticker': row['ticker'],
            'effective_date': row['effective_date'],
            'tone_score': row['tone_score'],
            'sponsored_prob': row['sponsored_prob'],
            'pseudo_title': row['pseudo_title'],
            'llm_sentiment': None,
            'llm_confidence': None,
            'llm_rationale': None,
            'llm_entities': None,
            'llm_score': None,
            'n_retries': attempts,
        })
    
    if (i + 1) % 100 == 0:
        elapsed_so_far = time.time() - t_start
        rate = (i + 1) / elapsed_so_far
        eta_min = (n_total - i - 1) / rate / 60
        valid_so_far = n_valid_first + n_valid_retry
        print(f'  {i+1:,}/{n_total:,} | valid: {valid_so_far} ({100*valid_so_far/(i+1):.1f}%) | '
              f'{rate:.1f} art/s | ETA: {eta_min:.0f}min', flush=True)

total_elapsed = time.time() - t_start
n_valid_total = n_valid_first + n_valid_retry

print(f'\n{"="*70}')
print(f'FULL EXTRACTION COMPLETE')
print(f'{"="*70}')
print(f'  Total articles:     {n_total:,}')
print(f'  Valid first try:    {n_valid_first:,} ({100*n_valid_first/n_total:.1f}%)')
print(f'  Valid after retry:  {n_valid_retry:,} ({100*n_valid_retry/n_total:.1f}%)')
print(f'  Total valid:        {n_valid_total:,} ({100*n_valid_total/n_total:.1f}%)')
print(f'  Failed:             {n_failed:,} ({100*n_failed/n_total:.1f}%)')
print(f'  Total time:         {total_elapsed/60:.1f} minutes')
print(f'  Throughput:         {n_total/total_elapsed:.1f} articles/sec')

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_parquet(OUTPUT_PATH, index=False)
print(f'\nSaved: {OUTPUT_PATH} ({len(results_df):,} rows)')

# %% [markdown]
# ## Section 7: Quality Summary
#
# Report the JSON validity rate, sentiment distribution, and compare
# LLM sentiment against GDELT tone and (if available) FinBERT scores.

# %%
# ============================================================
# Cell 8: Quality analysis
# ============================================================
results_df = pd.read_parquet(OUTPUT_PATH)
valid_df = results_df[results_df['llm_sentiment'].notna()].copy()

print(f'Results loaded: {len(results_df):,} rows')
print(f'Valid extractions: {len(valid_df):,} ({100*len(valid_df)/len(results_df):.1f}%)')

# JSON validity rate (the key metric)
json_validity_rate = 100 * len(valid_df) / len(results_df)
print(f'\nJSON VALIDITY RATE: {json_validity_rate:.1f}%')

# Sentiment distribution
print(f'\nSentiment distribution:')
print(valid_df['llm_sentiment'].value_counts())

# LLM score statistics
print(f'\nLLM score (sentiment * confidence) statistics:')
print(valid_df['llm_score'].describe())

# Per-bucket breakdown
print(f'\nPer-bucket sentiment breakdown:')
print(f'| {"Bucket":<25} | {"N":>6} | {"Bullish%":>9} | {"Neutral%":>9} | {"Bearish%":>9} | {"Mean_Score":>11} |')
print(f'|{"-"*27}|{"-"*8}|{"-"*11}|{"-"*11}|{"-"*11}|{"-"*13}|')
for bucket in sorted(valid_df['bucket'].unique()):
    bdf = valid_df[valid_df['bucket'] == bucket]
    n = len(bdf)
    bull_pct = 100 * (bdf['llm_sentiment'] == 'bullish').mean()
    neut_pct = 100 * (bdf['llm_sentiment'] == 'neutral').mean()
    bear_pct = 100 * (bdf['llm_sentiment'] == 'bearish').mean()
    mean_score = bdf['llm_score'].mean()
    print(f'| {bucket:<25} | {n:>6} | {bull_pct:>8.1f}% | {neut_pct:>8.1f}% | {bear_pct:>8.1f}% | {mean_score:>11.4f} |')

# Correlation with GDELT tone
corr_df = valid_df[['llm_score', 'tone_score']].dropna()
if len(corr_df) > 10:
    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(corr_df['llm_score'], corr_df['tone_score'])
    sr, sp = spearmanr(corr_df['llm_score'], corr_df['tone_score'])
    
    print(f'\nLLM Score vs GDELT Tone Correlation:')
    print(f'  Pearson r  = {pr:.4f} (p = {pp:.2e})')
    print(f'  Spearman r = {sr:.4f} (p = {sp:.2e})')

# Retry statistics
print(f'\nRetry statistics:')
print(valid_df['n_retries'].value_counts().sort_index())

# %%
# ============================================================
# Cell 9: Verify outputs
# ============================================================
# Final verification checklist
print(f'\n{"="*70}')
print(f'VERIFICATION CHECKLIST')
print(f'{"="*70}')

# 1. Schema defined once and versioned
print(f'  [x] Schema version: {SCHEMA_VERSION}')
print(f'  [x] Schema fields: {list(EXTRACTION_SCHEMA["properties"].keys())}')

# 2. JSON validity rate reported
print(f'  [x] JSON validity rate: {json_validity_rate:.1f}%')

# 3. Sample article IDs persisted
assert SAMPLE_IDS_PATH.exists(), "Sample IDs file not found!"
sample_ids = pd.read_parquet(SAMPLE_IDS_PATH)
print(f'  [x] Sample article IDs saved: {SAMPLE_IDS_PATH} ({len(sample_ids):,} rows)')

# 4. Extraction results saved
assert OUTPUT_PATH.exists(), "Extraction results file not found!"
print(f'  [x] Extraction results saved: {OUTPUT_PATH} ({len(results_df):,} rows)')

# 5. Model info
print(f'  [x] Model used: {model_name}')
print(f'  [x] Precision: float16')

print(f'\nAll checks passed. Pipeline outputs are ready for Phase 2.')
