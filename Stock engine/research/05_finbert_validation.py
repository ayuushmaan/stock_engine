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
# # 05 - FinBERT Sentiment Validation
#
# **Purpose**: Validate the GDELT tone-based signals used throughout this project
# against an independent NLP sentiment model (FinBERT). This serves as a
# robustness check: if GDELT tone and FinBERT sentiment are uncorrelated,
# our IC results may be an artefact of GDELT-specific noise rather than
# genuine market sentiment.
#
# **Design**:
# 1. Draw a stratified sample of ~5,000 articles from `sponsored_scores.parquet`,
#    proportionally across 4 buckets: Organic/Sponsored x Open/Closed window.
# 2. Extract pseudo-headlines from URL slugs (GDELT GKG does not store article text).
# 3. Score each pseudo-headline with `ProsusAI/finbert` sentiment model.
# 4. Report Pearson and Spearman correlation between FinBERT sentiment and GDELT tone.
# 5. Re-run IC calculations for Open-Window Organic vs Sponsored using FinBERT sentiment.
# 6. Compare FinBERT-based IC against established GDELT benchmarks (0.041 / 0.018).
#
# **Model choice**: `ProsusAI/finbert` is used instead of `yiyanghkust/finbert`
# because ProsusAI/finbert is the most widely cited FinBERT checkpoint for
# financial sentiment analysis (trained on Financial PhraseBank), produces
# calibrated positive/negative/neutral probabilities, and is actively maintained.
# Both are BERT-base models fine-tuned for financial text; results are comparable.

# %%
# ============================================================
# Cell 1: Setup and Imports
# ============================================================
import sys
import re
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

if sys.platform.startswith('win') and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / 'config' / 'settings.py').exists():
            return candidate
    raise FileNotFoundError('Could not locate project root containing config/settings.py')

ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_FINAL,
    DATA_PROCESSED,
    OUTPUTS_FIGURES,
    MIN_ARTICLES_FOR_SIGNAL,
    seed_everything,
)

seed_everything(42)

FIG_DIR = OUTPUTS_FIGURES / 'finbert_validation'
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})

print(f'Project root: {ROOT}')
print(f'Figure output: {FIG_DIR}')

# %% [markdown]
# ## Section 1: Stratified Sampling
#
# We draw ~5,000 articles proportionally from 4 buckets defined by:
# - **Source credibility**: Organic (`sponsored_prob < 0.3`) vs Sponsored (`sponsored_prob > 0.7`)
# - **Market timing**: Open (`time_bucket == "OPEN"`) vs Closed (`time_bucket in ["CLOSED_PRE", "CLOSED_POST"]`)
#
# Rows in the "uncertain" band (0.3 <= sponsored_prob <= 0.7) are excluded
# to match the clean separation used in the main analysis.

# %%
# ============================================================
# Cell 2: Load article-level data and create stratified sample
# ============================================================
TARGET_SAMPLE_SIZE = 5000

scores_df = pd.read_parquet(
    DATA_PROCESSED / 'sponsored_scores.parquet',
    columns=[
        'DocumentIdentifier', 'tone_score', 'sponsored_prob',
        'time_bucket', 'ticker', 'effective_date', 'source_domain',
    ]
)
print(f'Loaded sponsored_scores: {len(scores_df):,} rows')

# Classify source credibility
scores_df['credibility'] = np.where(
    scores_df['sponsored_prob'] < 0.3, 'organic',
    np.where(scores_df['sponsored_prob'] > 0.7, 'sponsored', 'uncertain')
)

# Classify market timing (collapse CLOSED_PRE + CLOSED_POST -> closed)
scores_df['window'] = np.where(
    scores_df['time_bucket'] == 'OPEN', 'open', 'closed'
)

# Exclude uncertain band
clean_df = scores_df[scores_df['credibility'] != 'uncertain'].copy()
print(f'After removing uncertain band: {len(clean_df):,} rows')

# Create bucket column
clean_df['bucket'] = clean_df['credibility'] + '_' + clean_df['window']

# Show bucket distribution
bucket_counts = clean_df['bucket'].value_counts()
print(f'\nBucket distribution (full population):')
for b, c in bucket_counts.items():
    print(f'  {b}: {c:,} ({100*c/len(clean_df):.1f}%)')

# Proportional stratified sample
np.random.seed(42)
bucket_proportions = bucket_counts / bucket_counts.sum()
sample_sizes = (bucket_proportions * TARGET_SAMPLE_SIZE).round().astype(int)

# Ensure total is exactly TARGET_SAMPLE_SIZE
diff = TARGET_SAMPLE_SIZE - sample_sizes.sum()
if diff != 0:
    # Adjust the largest bucket
    sample_sizes.iloc[0] += diff

sampled_dfs = []
for bucket_name, n_sample in sample_sizes.items():
    bucket_data = clean_df[clean_df['bucket'] == bucket_name]
    n_actual = min(n_sample, len(bucket_data))
    sampled = bucket_data.sample(n=n_actual, random_state=42)
    sampled_dfs.append(sampled)

sample_df = pd.concat(sampled_dfs, ignore_index=True)
print(f'\nStratified sample: {len(sample_df):,} articles')
print(f'Sample bucket distribution:')
for b, c in sample_df['bucket'].value_counts().items():
    print(f'  {b}: {c:,}')

# %% [markdown]
# ## Section 2: URL Slug Title Extraction
#
# GDELT GKG stores article URLs but not headlines. We extract pseudo-titles
# from URL slugs (e.g., `tata-motors-q3-revenue-growth` -> `tata motors q3 revenue growth`).
#
# This is an inherent limitation: URL slugs are noisier than real headlines.
# Articles with unparseable slugs (too short, purely numeric) are dropped.
# We report the parse success rate as a data quality metric.

# %%
# ============================================================
# Cell 3: Extract pseudo-titles from URL slugs
# ============================================================
def extract_title_from_url(url):
    """Extract a readable pseudo-title from URL path slug.
    
    Handles common Indian financial news URL patterns:
    - livemint.com/.../<slug>-<numeric_id>.html
    - economictimes.indiatimes.com/.../articleshow/<id>.cms
    - moneycontrol.com/news/.../<slug>-<id>.html
    """
    if not isinstance(url, str) or len(url) < 10:
        return None
    
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        
        # Remove common file extensions
        path = re.sub(r'\.(html?|cms|php|asp|aspx|jsp|xml|json)$', '', path, flags=re.IGNORECASE)
        
        # Get the last segment
        slug = path.split('/')[-1]
        
        # Handle economictimes "articleshow/<id>" pattern -> go up one level
        if slug.isdigit() or slug.lower() == 'articleshow':
            parts = path.split('/')
            for part in reversed(parts):
                if not part.isdigit() and part.lower() != 'articleshow' and len(part) > 3:
                    slug = part
                    break
        
        # Remove trailing numeric IDs (e.g., -11704567890, _12345)
        slug = re.sub(r'[-_]?\d{5,}$', '', slug)
        
        # Remove common prefixes that aren't meaningful
        slug = re.sub(r'^(article|story|news|post|blog)[-_]', '', slug, flags=re.IGNORECASE)
        
        # Convert hyphens/underscores to spaces
        title = re.sub(r'[-_]+', ' ', slug).strip()
        
        # Filter out too-short or purely numeric results
        if len(title) < 8 or title.replace(' ', '').isdigit():
            return None
        
        return title
    except Exception:
        return None


sample_df['pseudo_title'] = sample_df['DocumentIdentifier'].apply(extract_title_from_url)

parse_success = sample_df['pseudo_title'].notna().sum()
parse_rate = 100 * parse_success / len(sample_df)
print(f'URL parse success: {parse_success:,} / {len(sample_df):,} ({parse_rate:.1f}%)')

# Drop rows where title extraction failed
sample_df = sample_df[sample_df['pseudo_title'].notna()].copy()
print(f'Sample after dropping unparseable URLs: {len(sample_df):,}')

# Show examples
print('\nSample pseudo-titles:')
for _, row in sample_df.head(10).iterrows():
    print(f'  [{row["bucket"]}] {row["pseudo_title"][:80]}')

# %% [markdown]
# ## Section 3: FinBERT Sentiment Scoring
#
# We use `ProsusAI/finbert` to score each pseudo-title. The model outputs
# probabilities for [positive, negative, neutral]. We compute:
# - **finbert_score**: P(positive) - P(negative), a continuous sentiment score in [-1, 1]
# - **finbert_label**: argmax class label
#
# Processing is batched for efficiency.

# %%
# ============================================================
# Cell 4: Load FinBERT model and score articles
# ============================================================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

print(f'Loading {MODEL_NAME}...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f'Model loaded on device: {device}')
print(f'Label mapping: {model.config.id2label}')

# Batch inference
BATCH_SIZE = 64
texts = sample_df['pseudo_title'].tolist()
all_probs = []

print(f'Scoring {len(texts):,} articles in batches of {BATCH_SIZE}...')
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]
    
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    
    if (i // BATCH_SIZE) % 20 == 0:
        print(f'  Processed {min(i + BATCH_SIZE, len(texts)):,} / {len(texts):,}')

all_probs = np.vstack(all_probs)
print(f'Scoring complete. Output shape: {all_probs.shape}')

# Map model labels to columns
# ProsusAI/finbert: {0: 'positive', 1: 'negative', 2: 'neutral'}
label_map = model.config.id2label
label_cols = {v: i for i, v in label_map.items()}

sample_df['finbert_positive'] = all_probs[:, label_cols['positive']]
sample_df['finbert_negative'] = all_probs[:, label_cols['negative']]
sample_df['finbert_neutral'] = all_probs[:, label_cols['neutral']]

# Continuous score: P(positive) - P(negative)
sample_df['finbert_score'] = sample_df['finbert_positive'] - sample_df['finbert_negative']

# Discrete label
sample_df['finbert_label'] = np.array([label_map[i] for i in all_probs.argmax(axis=1)])

print(f'\nFinBERT label distribution:')
print(sample_df['finbert_label'].value_counts())
print(f'\nFinBERT score statistics:')
print(sample_df['finbert_score'].describe())

# %% [markdown]
# ## Section 4: FinBERT vs GDELT Tone Correlation
#
# We compute Pearson and Spearman correlation between `finbert_score` and
# `tone_score` (GDELT). High correlation validates that GDELT tone captures
# similar sentiment as a transformer-based model. Low correlation would
# suggest our IC results might be specific to GDELT's tone measure.

# %%
# ============================================================
# Cell 5: Correlation analysis
# ============================================================
# Drop rows with missing tone
corr_df = sample_df[['finbert_score', 'tone_score', 'bucket', 'credibility', 'window']].dropna()
print(f'Rows with both finbert_score and tone_score: {len(corr_df):,}')

# Overall correlation
pearson_r, pearson_p = pearsonr(corr_df['finbert_score'], corr_df['tone_score'])
spearman_r, spearman_p = spearmanr(corr_df['finbert_score'], corr_df['tone_score'])

print(f'\n{"="*70}')
print(f'OVERALL FINBERT vs GDELT TONE CORRELATION')
print(f'{"="*70}')
print(f'| {"Metric":<15} | {"Value":>10} | {"p-value":>12} |')
print(f'|{"-"*17}|{"-"*12}|{"-"*14}|')
print(f'| {"Pearson r":<15} | {pearson_r:>10.4f} | {pearson_p:>12.2e} |')
print(f'| {"Spearman rho":<15} | {spearman_r:>10.4f} | {spearman_p:>12.2e} |')
print(f'{"="*70}')

# Per-bucket correlation
print(f'\nPer-bucket correlation:')
print(f'| {"Bucket":<25} | {"N":>6} | {"Pearson_r":>10} | {"Spearman_r":>10} |')
print(f'|{"-"*27}|{"-"*8}|{"-"*12}|{"-"*12}|')
for bucket in sorted(corr_df['bucket'].unique()):
    bdf = corr_df[corr_df['bucket'] == bucket]
    pr, _ = pearsonr(bdf['finbert_score'], bdf['tone_score'])
    sr, _ = spearmanr(bdf['finbert_score'], bdf['tone_score'])
    print(f'| {bucket:<25} | {len(bdf):>6} | {pr:>10.4f} | {sr:>10.4f} |')

# %%
# ============================================================
# Cell 6: Correlation scatter plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: overall scatter
ax = axes[0]
ax.scatter(corr_df['tone_score'], corr_df['finbert_score'],
           alpha=0.15, s=8, color='#2563eb')
ax.set_xlabel('GDELT Tone Score')
ax.set_ylabel('FinBERT Score (P(pos) - P(neg))')
ax.set_title(f'Overall: Pearson r = {pearson_r:.3f}', fontweight='bold')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.grid(True, alpha=0.2)

# Add regression line
z = np.polyfit(corr_df['tone_score'].values, corr_df['finbert_score'].values, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(corr_df['tone_score'].min(), corr_df['tone_score'].max(), 100)
ax.plot(x_line, p_fit(x_line), color='#dc2626', lw=2, label=f'OLS fit (slope={z[0]:.3f})')
ax.legend()

# Right: by bucket
ax = axes[1]
bucket_colors = {
    'organic_closed': '#2563eb',
    'organic_open': '#60a5fa',
    'sponsored_closed': '#dc2626',
    'sponsored_open': '#f87171',
}
for bucket, color in bucket_colors.items():
    bdf = corr_df[corr_df['bucket'] == bucket]
    ax.scatter(bdf['tone_score'], bdf['finbert_score'],
               alpha=0.2, s=8, color=color, label=bucket)
ax.set_xlabel('GDELT Tone Score')
ax.set_ylabel('FinBERT Score')
ax.set_title('By Bucket', fontweight='bold')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

fig.suptitle('FinBERT Sentiment vs GDELT Tone Score', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'finbert_vs_gdelt_scatter.png')
plt.close(fig)
print(f'Saved: {FIG_DIR / "finbert_vs_gdelt_scatter.png"}')

# %% [markdown]
# ## Section 5: FinBERT Sentiment Distribution by Bucket
#
# Compare the sentiment distributions across the 4 buckets.
# If the classifier is truly separating organic from sponsored content,
# we might expect systematic differences in FinBERT sentiment.

# %%
# ============================================================
# Cell 7: Sentiment distribution by bucket
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: violin plot of finbert_score by credibility
ax = axes[0]
organic_scores = sample_df[sample_df['credibility'] == 'organic']['finbert_score'].dropna()
sponsored_scores = sample_df[sample_df['credibility'] == 'sponsored']['finbert_score'].dropna()

parts = ax.violinplot([organic_scores.values, sponsored_scores.values],
                       positions=[0, 1], showmeans=True, showmedians=True)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Organic', 'Sponsored'])
ax.set_ylabel('FinBERT Score')
ax.set_title('FinBERT Sentiment by Source Credibility', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# Add mean annotations
ax.annotate(f'mean={organic_scores.mean():.3f}', xy=(0, organic_scores.mean()),
            xytext=(0.3, organic_scores.mean() + 0.1),
            fontsize=10, color='#2563eb', fontweight='bold')
ax.annotate(f'mean={sponsored_scores.mean():.3f}', xy=(1, sponsored_scores.mean()),
            xytext=(1.1, sponsored_scores.mean() + 0.1),
            fontsize=10, color='#dc2626', fontweight='bold')

# Right: by window
ax = axes[1]
open_scores = sample_df[sample_df['window'] == 'open']['finbert_score'].dropna()
closed_scores = sample_df[sample_df['window'] == 'closed']['finbert_score'].dropna()

parts = ax.violinplot([open_scores.values, closed_scores.values],
                       positions=[0, 1], showmeans=True, showmedians=True)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Open Window', 'Closed Window'])
ax.set_ylabel('FinBERT Score')
ax.set_title('FinBERT Sentiment by Market Window', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('FinBERT Sentiment Distribution Across Buckets', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'finbert_distribution_by_bucket.png')
plt.close(fig)
print(f'Saved: {FIG_DIR / "finbert_distribution_by_bucket.png"}')

# Print summary stats
print(f'\nFinBERT Score Summary by Bucket:')
print(f'| {"Bucket":<25} | {"N":>6} | {"Mean":>8} | {"Std":>8} | {"Median":>8} |')
print(f'|{"-"*27}|{"-"*8}|{"-"*10}|{"-"*10}|{"-"*10}|')
for bucket in sorted(sample_df['bucket'].unique()):
    bdf = sample_df[sample_df['bucket'] == bucket]['finbert_score'].dropna()
    print(f'| {bucket:<25} | {len(bdf):>6} | {bdf.mean():>8.4f} | {bdf.std():>8.4f} | {bdf.median():>8.4f} |')

# %% [markdown]
# ## Section 6: IC Re-calculation Using FinBERT Sentiment
#
# We re-compute IC for the Open-Window signal using FinBERT sentiment
# instead of GDELT tone. This is the key test:
#
# - From H1 & H2, Open-Window Sponsored IC ~ 0.041, Open-Window Organic IC ~ 0.018
# - If FinBERT-based IC shows the **same sign and similar magnitude**, the
#   signal is robust to sentiment measure choice.
# - If the IC **flips sign or collapses**, it flags a potential pipeline/join error
#   or that the signal is a GDELT-specific artefact.
#
# We aggregate FinBERT scores to daily ticker-level signals, then compute
# cross-sectional IC against same-day intraday returns (open-window alignment).

# %%
# ============================================================
# Cell 8: Aggregate FinBERT to daily signals and compute IC
# ============================================================
# Filter to open-window articles only
open_sample = sample_df[sample_df['window'] == 'open'].copy()
open_sample['effective_date'] = pd.to_datetime(open_sample['effective_date'])

print(f'Open-window articles in sample: {len(open_sample):,}')

# Aggregate FinBERT score to daily-ticker level (mean)
finbert_daily = (
    open_sample
    .groupby(['effective_date', 'ticker', 'credibility'])
    .agg(
        finbert_signal=('finbert_score', 'mean'),
        gdelt_signal=('tone_score', 'mean'),
        n_articles=('finbert_score', 'count'),
    )
    .reset_index()
)

print(f'Daily-ticker aggregates: {len(finbert_daily):,} rows')

# Load returns for merging
master_df = pd.read_parquet(DATA_FINAL / 'master_dataset.parquet',
                            columns=['ticker', 'date', 'ret_intraday', 'ret_close2close'])
master_df['date'] = pd.to_datetime(master_df['date'])

# Merge FinBERT signals with returns
finbert_daily = finbert_daily.merge(
    master_df,
    left_on=['effective_date', 'ticker'],
    right_on=['date', 'ticker'],
    how='inner'
)

print(f'After merging with returns: {len(finbert_daily):,} rows')

# Compute daily IC function (reused from 02_hypothesis_h1)
def compute_daily_ic(df, signal_col, return_col, min_stocks=5):
    """Compute daily cross-sectional IC (Spearman correlation)."""
    ic_list = []
    dates = []
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        valid_mask = day_data[[signal_col, return_col]].notna().all(axis=1)
        if valid_mask.sum() < min_stocks:
            continue
        day_valid = day_data[valid_mask]
        ic, _ = spearmanr(day_valid[signal_col].values, day_valid[return_col].values)
        ic_list.append(ic)
        dates.append(date)
    return np.array(ic_list), np.array(dates)

# Split by credibility and compute IC
results_ic = {}
for cred in ['organic', 'sponsored']:
    cred_df = finbert_daily[finbert_daily['credibility'] == cred]
    
    if len(cred_df) < 10:
        print(f'\n  {cred}: Too few rows ({len(cred_df)}) for IC computation')
        continue
    
    # FinBERT-based IC
    ic_fb, dates_fb = compute_daily_ic(cred_df, 'finbert_signal', 'ret_intraday', min_stocks=3)
    
    # GDELT-based IC (on same sample for apples-to-apples comparison)
    ic_gd, dates_gd = compute_daily_ic(cred_df, 'gdelt_signal', 'ret_intraday', min_stocks=3)
    
    if len(ic_fb) > 1:
        fb_mean = np.nanmean(ic_fb)
        fb_std = np.nanstd(ic_fb, ddof=1)
        fb_ir = fb_mean / fb_std if fb_std > 0 else np.nan
        t_fb, p_fb = ttest_1samp(ic_fb, 0, nan_policy='omit')
    else:
        fb_mean = fb_std = fb_ir = t_fb = p_fb = np.nan
    
    if len(ic_gd) > 1:
        gd_mean = np.nanmean(ic_gd)
        gd_std = np.nanstd(ic_gd, ddof=1)
        gd_ir = gd_mean / gd_std if gd_std > 0 else np.nan
        t_gd, p_gd = ttest_1samp(ic_gd, 0, nan_policy='omit')
    else:
        gd_mean = gd_std = gd_ir = t_gd = p_gd = np.nan
    
    results_ic[cred] = {
        'finbert': {'mean': fb_mean, 'std': fb_std, 'ir': fb_ir, 't': t_fb, 'p': p_fb, 'n_days': len(ic_fb)},
        'gdelt': {'mean': gd_mean, 'std': gd_std, 'ir': gd_ir, 't': t_gd, 'p': p_gd, 'n_days': len(ic_gd)},
    }

# Print IC comparison table
print(f'\n{"="*95}')
print(f'IC COMPARISON: FinBERT vs GDELT (Open-Window, Same Sample)')
print(f'{"="*95}')
print(f'| {"Signal":<22} | {"N_days":>6} | {"IC_mean":>10} | {"IC_std":>10} | {"ICIR":>8} | {"t_stat":>8} | {"p_value":>10} |')
print(f'|{"-"*24}|{"-"*8}|{"-"*12}|{"-"*12}|{"-"*10}|{"-"*10}|{"-"*12}|')

for cred in ['organic', 'sponsored']:
    if cred not in results_ic:
        continue
    for measure in ['finbert', 'gdelt']:
        r = results_ic[cred][measure]
        label = f'{cred.capitalize()} ({measure})'
        print(f'| {label:<22} | {r["n_days"]:>6} | {r["mean"]:>10.6f} | {r["std"]:>10.6f} | {r["ir"]:>8.4f} | {r["t"]:>8.4f} | {r["p"]:>10.6f} |')
    print(f'|{"-"*24}|{"-"*8}|{"-"*12}|{"-"*12}|{"-"*10}|{"-"*10}|{"-"*12}|')

print(f'{"="*95}')

# %% [markdown]
# ## Section 7: Benchmark Comparison
#
# Compare the FinBERT-based IC against the full-sample GDELT benchmarks
# established in notebooks 02 and 04:
# - **Sponsored Open-Window IC**: ~0.041 (GDELT full sample)
# - **Organic Open-Window IC**: ~0.018 (GDELT full sample)
#
# A large discrepancy in sign or order of magnitude is flagged.

# %%
# ============================================================
# Cell 9: Benchmark comparison and flag check
# ============================================================
BENCHMARK_SPONSORED_IC = 0.041
BENCHMARK_ORGANIC_IC = 0.018

print(f'\n{"="*80}')
print(f'BENCHMARK COMPARISON')
print(f'{"="*80}')
print(f'| {"Metric":<30} | {"GDELT Benchmark":>16} | {"FinBERT (this sample)":>22} | {"Flag":>8} |')
print(f'|{"-"*32}|{"-"*18}|{"-"*24}|{"-"*10}|')

for cred, benchmark in [('organic', BENCHMARK_ORGANIC_IC), ('sponsored', BENCHMARK_SPONSORED_IC)]:
    if cred in results_ic:
        fb_ic = results_ic[cred]['finbert']['mean']
        
        # Flag if sign differs or magnitude differs by > 3x
        sign_match = (np.sign(fb_ic) == np.sign(benchmark)) if not (np.isnan(fb_ic) or benchmark == 0) else True
        magnitude_ok = abs(fb_ic) < 3 * abs(benchmark) if benchmark != 0 else True
        
        if np.isnan(fb_ic):
            flag = 'N/A'
        elif not sign_match:
            flag = 'SIGN!'
        elif not magnitude_ok:
            flag = 'MAG!'
        else:
            flag = 'OK'
        
        label = f'{cred.capitalize()} Open-Window IC'
        print(f'| {label:<30} | {benchmark:>16.4f} | {fb_ic:>22.6f} | {flag:>8} |')

print(f'{"="*80}')

# Interpretation
print('\nInterpretation:')
print('  OK    = FinBERT IC is consistent with GDELT benchmark (same sign, similar magnitude)')
print('  SIGN! = WARNING: FinBERT IC has opposite sign - potential pipeline/join error')
print('  MAG!  = CAUTION: FinBERT IC magnitude differs >3x - may reflect URL-slug noise')
print('  N/A   = Insufficient data for computation')

# %% [markdown]
# ## Section 8: Limitations and Caveats
#
# **Critical limitations of this validation**:
#
# 1. **URL slugs are not headlines**: We are using URL path fragments as a proxy
#    for article headlines. This introduces noise: some slugs are truncated,
#    others contain non-English text (Hindi financial news sites), and some
#    URLs have opaque ID-based paths with no readable slug.
#
# 2. **Sample size for IC**: With ~5,000 articles split across 4 buckets and
#    ~50 tickers, daily cross-sections are thin. The IC calculation requires
#    >= 3 stocks per day, which limits the number of valid IC days.
#
# 3. **Model domain mismatch**: ProsusAI/finbert is trained on English financial
#    text (Financial PhraseBank). Indian financial news may include Hindi/regional
#    language content that the model cannot process.
#
# 4. **Same-sample IC caveat**: The IC comparison in Section 6 uses the
#    same ~5,000 article subsample, not the full dataset. The GDELT benchmarks
#    (0.041 / 0.018) are computed on the entire dataset; direct magnitude
#    comparison should account for sampling noise.

# %%
# ============================================================
# Cell 10: Final summary
# ============================================================
print(f'\n{"="*80}')
print(f'FINBERT VALIDATION SUMMARY')
print(f'{"="*80}')
print(f'  Total articles sampled:     {len(sample_df):,}')
print(f'  URL parse success rate:     {parse_rate:.1f}%')
print(f'  FinBERT model:              {MODEL_NAME}')
print(f'  Device:                     {device}')
print(f'')
print(f'  Overall FinBERT-GDELT correlation:')
print(f'    Pearson r  = {pearson_r:.4f} (p = {pearson_p:.2e})')
print(f'    Spearman r = {spearman_r:.4f} (p = {spearman_p:.2e})')
print(f'')

if pearson_r > 0.3:
    print(f'  CONCLUSION: Moderate-to-strong correlation between FinBERT and GDELT tone.')
    print(f'  The GDELT tone-based signals used in H1/H2 are validated by an independent')
    print(f'  transformer-based sentiment model. IC results are unlikely to be GDELT-specific.')
elif pearson_r > 0.1:
    print(f'  CONCLUSION: Weak-to-moderate correlation between FinBERT and GDELT tone.')
    print(f'  The two measures partially agree. IC results should be interpreted with the')
    print(f'  caveat that GDELT tone captures different aspects than FinBERT sentiment.')
    print(f'  URL-slug noise likely depresses the observed correlation.')
else:
    print(f'  CONCLUSION: Low correlation between FinBERT and GDELT tone.')
    print(f'  This could indicate: (a) URL slugs are too noisy for FinBERT,')
    print(f'  (b) GDELT tone measures something fundamentally different, or')
    print(f'  (c) the IC results may be GDELT-specific. Further investigation needed.')

print(f'{"="*80}')
