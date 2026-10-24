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
# # H1: Organic vs Sponsored News Signals
#
# **Hypothesis**: Organic/editorial news carries stronger return-predictive signal than sponsored/PR content.

# %%
# Setup: Imports and config
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
    NUMPY_SEED,
    seed_everything,
)

seed_everything(42)  # Override with 42 per requirements

FIG_DIR_H1 = OUTPUTS_FIGURES / 'h1'
FIG_DIR_H1.mkdir(parents=True, exist_ok=True)

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
print(f'Figure output: {FIG_DIR_H1}')

# %% [markdown]
# ## Section 1: Signal Construction
#
# Load master_dataset, filter to rows with article_count_total >= 3, then construct normalized organic and sponsored signals.

# %%
# Load and filter master dataset
MASTER_PATH = DATA_FINAL / 'master_dataset.parquet'
master_df = pd.read_parquet(MASTER_PATH)

master_df['date'] = pd.to_datetime(master_df['date'])
master_df = master_df.sort_values(['date', 'ticker']).reset_index(drop=True)

# Filter to rows with article_count_total >= 3
filtered_df = master_df[master_df['article_count_total'] >= MIN_ARTICLES_FOR_SIGNAL].copy()
print(f'Loaded {len(master_df):,} rows, filtered to {len(filtered_df):,} rows (article_count_total >= {MIN_ARTICLES_FOR_SIGNAL})')

# Construct signals using nanmean
filtered_df['organic_signal'] = np.nanmean(
    filtered_df[['signal_organic_closed', 'signal_organic_open']].values,
    axis=1
)
filtered_df['sponsored_signal'] = np.nanmean(
    filtered_df[['signal_sponsored_closed', 'signal_sponsored_open']].values,
    axis=1
)

# Define train split: 2020-2023
train_mask = (filtered_df['date'] >= '2020-01-01') & (filtered_df['date'] < '2024-01-01')
train_df = filtered_df[train_mask]

print(f'Train split (2020-2023): {len(train_df):,} rows')
print(f'Out-of-sample / Holdout (2024-07+): {len(filtered_df[filtered_df["date"] >= "2024-07-01"]):,} rows')

# Compute scaler on train split only
organic_mean = train_df['organic_signal'].mean()
organic_std = train_df['organic_signal'].std()
sponsored_mean = train_df['sponsored_signal'].mean()
sponsored_std = train_df['sponsored_signal'].std()

print(f'\nTrain split normalization:')
print(f'  Organic:  mean={organic_mean:.6f}, std={organic_std:.6f}')
print(f'  Sponsored: mean={sponsored_mean:.6f}, std={sponsored_std:.6f}')

# Normalize both signals (apply train scaler to all rows)
filtered_df['organic_signal_norm'] = (filtered_df['organic_signal'] - organic_mean) / organic_std
filtered_df['sponsored_signal_norm'] = (filtered_df['sponsored_signal'] - sponsored_mean) / sponsored_std

print(f'\nSignals normalized and ready for analysis.')
print(f'Rows with valid normalized organic signal: {filtered_df["organic_signal_norm"].notna().sum():,}')
print(f'Rows with valid normalized sponsored signal: {filtered_df["sponsored_signal_norm"].notna().sum():,}')


# %% [markdown]
# ## Section 2: IC Test
#
# Compute daily cross-sectional IC (Spearman correlation between signal and next-day returns). Bootstrap for confidence intervals.

# %%
# Function to compute daily cross-sectional IC
def compute_daily_ic(df, signal_col, return_col):
    """
    Compute daily cross-sectional IC (Spearman correlation).
    Only include days with >= 5 valid stocks.
    """
    ic_list = []
    dates = []
    
    for date in df['date'].unique():
        day_data = df[df['date'] == date].copy()
        
        # Check if we have >= 5 valid observations
        valid_mask = day_data[[signal_col, return_col]].notna().all(axis=1)
        if valid_mask.sum() < 5:
            continue
        
        day_valid = day_data[valid_mask]
        signal = day_valid[signal_col].values
        returns = day_valid[return_col].values
        
        # Compute Spearman IC
        ic, _ = spearmanr(signal, returns)
        ic_list.append(ic)
        dates.append(date)
    
    return np.array(ic_list), np.array(dates)

# Compute IC for both signals
# Note: ret_close2close is next-day return, so we shift when matching
# We compute IC as correlation(signal_today, return_tomorrow)

ic_organic, dates_organic = compute_daily_ic(
    filtered_df, 'organic_signal_norm', 'ret_close2close'
)
ic_sponsored, dates_sponsored = compute_daily_ic(
    filtered_df, 'sponsored_signal_norm', 'ret_close2close'
)

print(f'Organic IC days: {len(ic_organic)}')
print(f'Sponsored IC days: {len(ic_sponsored)}')

# Aggregate statistics for organic signal
ic_org_mean = np.nanmean(ic_organic)
ic_org_std = np.nanstd(ic_organic, ddof=1)
ic_org_ir = ic_org_mean / ic_org_std if ic_org_std > 0 else np.nan
t_org, p_org = ttest_1samp(ic_organic, 0, nan_policy='omit')

# Aggregate statistics for sponsored signal
ic_sp_mean = np.nanmean(ic_sponsored)
ic_sp_std = np.nanstd(ic_sponsored, ddof=1)
ic_sp_ir = ic_sp_mean / ic_sp_std if ic_sp_std > 0 else np.nan
t_sp, p_sp = ttest_1samp(ic_sponsored, 0, nan_policy='omit')

# Bootstrap for confidence intervals (1000 samples)
n_bootstrap = 1000
np.random.seed(42)

bootstrap_org_means = []
bootstrap_sp_means = []
bootstrap_diff_means = []

for _ in range(n_bootstrap):
    # Resample days with replacement
    idx = np.random.choice(len(ic_organic), size=len(ic_organic), replace=True)
    bootstrap_org_means.append(np.nanmean(ic_organic[idx]))
    
    idx = np.random.choice(len(ic_sponsored), size=len(ic_sponsored), replace=True)
    bootstrap_sp_means.append(np.nanmean(ic_sponsored[idx]))
    
    # For difference, resample the same indices for pairing
    idx = np.random.choice(min(len(ic_organic), len(ic_sponsored)), 
                           size=min(len(ic_organic), len(ic_sponsored)), replace=True)
    diff = ic_organic[:min(len(ic_organic), len(ic_sponsored))][idx] - \
           ic_sponsored[:min(len(ic_organic), len(ic_sponsored))][idx]
    bootstrap_diff_means.append(np.nanmean(diff))

bootstrap_org_means = np.array(bootstrap_org_means)
bootstrap_sp_means = np.array(bootstrap_sp_means)
bootstrap_diff_means = np.array(bootstrap_diff_means)

ci_org_low = np.percentile(bootstrap_org_means, 2.5)
ci_org_high = np.percentile(bootstrap_org_means, 97.5)
ci_sp_low = np.percentile(bootstrap_sp_means, 2.5)
ci_sp_high = np.percentile(bootstrap_sp_means, 97.5)
ci_diff_low = np.percentile(bootstrap_diff_means, 2.5)
ci_diff_high = np.percentile(bootstrap_diff_means, 97.5)

# Paired difference statistics
diff_ic = ic_organic[:min(len(ic_organic), len(ic_sponsored))] - \
          ic_sponsored[:min(len(ic_organic), len(ic_sponsored))]
diff_mean = np.nanmean(diff_ic)
diff_std = np.nanstd(diff_ic, ddof=1)
t_diff, p_diff = ttest_1samp(diff_ic, 0, nan_policy='omit')

# Print results table
print('\n' + '='*95)
print('IC TEST RESULTS')
print('='*95)
print(f'| {"Signal":<12} | {"N_days":>6} | {"IC_mean":>10} | {"IC_std":>10} | {"ICIR":>8} | {"t_stat":>8} | {"p_value":>9} | {"CI_low":>10} | {"CI_high":>10} |')
print('|' + '-'*93 + '|')
print(f'| {"Organic":<12} | {len(ic_organic):>6} | {ic_org_mean:>10.6f} | {ic_org_std:>10.6f} | {ic_org_ir:>8.4f} | {t_org:>8.4f} | {p_org:>9.6f} | {ci_org_low:>10.6f} | {ci_org_high:>10.6f} |')
print(f'| {"Sponsored":<12} | {len(ic_sponsored):>6} | {ic_sp_mean:>10.6f} | {ic_sp_std:>10.6f} | {ic_sp_ir:>8.4f} | {t_sp:>8.4f} | {p_sp:>9.6f} | {ci_sp_low:>10.6f} | {ci_sp_high:>10.6f} |')
print(f'| {"Difference":<12} | {len(diff_ic):>6} | {diff_mean:>10.6f} | {diff_std:>10.6f} | {"N/A":>8} | {t_diff:>8.4f} | {p_diff:>9.6f} | {ci_diff_low:>10.6f} | {ci_diff_high:>10.6f} |')
print('='*95)

# %% [markdown]
# ## Section 2b: Extended IC Analysis (Train Split)
#
# Compute additional metrics for organic and sponsored signals on the train split:
# 1. ICIR (IC_mean / IC_std)
# 2. IC Hit Rate (% of days where IC_t > 0)
# 3. IC Stability (% of rolling 60-day windows where rolling_IC > 0)
# 4. Tail behavior (Mean IC on days when |IC| > 0.05 and % of strong signal days)
# 5. Correlation between organic_IC_t and sponsored_IC_t per day

# %%
# Compute daily IC on train split specifically
train_df_norm = filtered_df[train_mask]
ic_org_train, dates_org_train = compute_daily_ic(train_df_norm, 'organic_signal_norm', 'ret_close2close')
ic_sp_train, dates_sp_train = compute_daily_ic(train_df_norm, 'sponsored_signal_norm', 'ret_close2close')

df_ic_train = pd.DataFrame({
    'date': dates_org_train,
    'ic_organic': ic_org_train
}).merge(
    pd.DataFrame({
        'date': dates_sp_train,
        'ic_sponsored': ic_sp_train
    }),
    on='date',
    how='inner'
).sort_values('date').reset_index(drop=True)

ic_org_train_aligned = df_ic_train['ic_organic'].values
ic_sp_train_aligned = df_ic_train['ic_sponsored'].values

# 1. IC mean and ICIR
ic_org_mean_train = np.mean(ic_org_train_aligned)
ic_org_std_train = np.std(ic_org_train_aligned, ddof=1)
ic_org_ir_train = ic_org_mean_train / ic_org_std_train if ic_org_std_train > 0 else np.nan

ic_sp_mean_train = np.mean(ic_sp_train_aligned)
ic_sp_std_train = np.std(ic_sp_train_aligned, ddof=1)
ic_sp_ir_train = ic_sp_mean_train / ic_sp_std_train if ic_sp_std_train > 0 else np.nan

# 2. IC Hit Rate
hit_rate_org = np.mean(ic_org_train_aligned > 0) * 100
hit_rate_sp = np.mean(ic_sp_train_aligned > 0) * 100

# 3. IC Stability (rolling 60-day windows rolling_IC > 0)
rolling_ic_org = df_ic_train['ic_organic'].rolling(window=60, min_periods=1).mean()
stability_org = np.mean(rolling_ic_org > 0) * 100

rolling_ic_sp = df_ic_train['ic_sponsored'].rolling(window=60, min_periods=1).mean()
stability_sp = np.mean(rolling_ic_sp > 0) * 100

# 4. Tail behavior
strong_mask_org = np.abs(ic_org_train_aligned) > 0.05
strong_days_pct_org = np.mean(strong_mask_org) * 100
mean_ic_strong_org = np.mean(ic_org_train_aligned[strong_mask_org]) if np.sum(strong_mask_org) > 0 else np.nan

strong_mask_sp = np.abs(ic_sp_train_aligned) > 0.05
strong_days_pct_sp = np.mean(strong_mask_sp) * 100
mean_ic_strong_sp = np.mean(ic_sp_train_aligned[strong_mask_sp]) if np.sum(strong_mask_sp) > 0 else np.nan

# 5. Correlation & Cross-correlation
pearson_corr = df_ic_train['ic_organic'].corr(df_ic_train['ic_sponsored'], method='pearson')
spearman_corr = df_ic_train['ic_organic'].corr(df_ic_train['ic_sponsored'], method='spearman')

# Print the extended table
print('\nEXTENDED IC ANALYSIS (TRAIN SPLIT)')
print('| Signal    | IC_mean | ICIR | Hit_Rate | Stability | Strong_Days% | Corr_with_other |')
print('|-----------|---------|------|----------|-----------|--------------|-----------------|')
print(f'| Organic   | {ic_org_mean_train:.6f} | {ic_org_ir_train:.4f} | {hit_rate_org:.2f}% | {stability_org:.2f}% | {strong_days_pct_org:.2f}% | {pearson_corr:.6f} |')
print(f'| Sponsored | {ic_sp_mean_train:.6f} | {ic_sp_ir_train:.4f} | {hit_rate_sp:.2f}% | {stability_sp:.2f}% | {strong_days_pct_sp:.2f}% | {pearson_corr:.6f} |')

print(f'\nTail Behavior (Mean IC on strong days, |IC| > 0.05):')
print(f'  Organic Strong Days Mean IC: {mean_ic_strong_org:.6f}')
print(f'  Sponsored Strong Days Mean IC: {mean_ic_strong_sp:.6f}')
print(f'\nCross-correlation between daily organic IC and daily sponsored IC series:')
print(f'  Pearson correlation: {pearson_corr:.6f}')
print(f'  Spearman correlation: {spearman_corr:.6f}')

# Store results for later use
results_train = {
    'ic_organic_mean': ic_org_mean,
    'ic_organic_std': ic_org_std,
    'ic_organic_ir': ic_org_ir,
    't_organic': t_org,
    'p_organic': p_org,
    'ic_sponsored_mean': ic_sp_mean,
    'ic_sponsored_std': ic_sp_std,
    'ic_sponsored_ir': ic_sp_ir,
    't_sponsored': t_sp,
    'p_sponsored': p_sp,
    'diff_mean': diff_mean,
    't_diff': t_diff,
    'p_diff': p_diff,
    'ci_org_low': ci_org_low,
    'ci_org_high': ci_org_high,
    'ci_sp_low': ci_sp_low,
    'ci_sp_high': ci_sp_high,
    'ci_diff_low': ci_diff_low,
    'ci_diff_high': ci_diff_high,
}

# %% [markdown]
# ## Section 3: Rolling IC Plot
#
# 60-day rolling IC with COVID and holdout period shading.

# %%
# Create IC time series dataframes
ic_org_ts = pd.DataFrame({
    'date': dates_organic,
    'ic': ic_organic
}).sort_values('date').reset_index(drop=True)

ic_sp_ts = pd.DataFrame({
    'date': dates_sponsored,
    'ic': ic_sponsored
}).sort_values('date').reset_index(drop=True)

# Compute 60-day rolling average
ic_org_ts['rolling_ic_60'] = ic_org_ts['ic'].rolling(window=60, min_periods=1).mean()
ic_sp_ts['rolling_ic_60'] = ic_sp_ts['ic'].rolling(window=60, min_periods=1).mean()

# Merge on date for plotting
ic_rolling = ic_org_ts[['date', 'rolling_ic_60']].rename(columns={'rolling_ic_60': 'organic'}).merge(
    ic_sp_ts[['date', 'rolling_ic_60']].rename(columns={'rolling_ic_60': 'sponsored'}),
    on='date',
    how='outer'
)

# Create plot
fig, ax = plt.subplots(figsize=(18, 7))

# Shade COVID period (2020-03 to 2020-06)
ax.axvspan(
    pd.Timestamp('2020-03-01'),
    pd.Timestamp('2020-07-01'),
    color='#fca5a5',
    alpha=0.25,
    label='COVID'
)

# Shade Holdout period (2024-07 onwards)
ax.axvspan(
    pd.Timestamp('2024-07-01'),
    ic_rolling['date'].max(),
    color='#d1d5db',
    alpha=0.25,
    label='Holdout'
)

# Plot rolling ICs
ax.plot(ic_rolling['date'], ic_rolling['organic'], color='#2563eb', linewidth=2, label='Organic (60-day rolling)')
ax.plot(ic_rolling['date'], ic_rolling['sponsored'], color='#dc2626', linewidth=2, linestyle='--', label='Sponsored (60-day rolling)')

# Horizontal line at IC=0
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

ax.set_title('Rolling 60-Day IC: Organic vs Sponsored News Signal', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Information Coefficient (Spearman)', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()
fig.tight_layout()

rolling_ic_path = FIG_DIR_H1 / 'rolling_ic_organic_vs_sponsored.png'
fig.savefig(rolling_ic_path)
plt.close(fig)
print(f'Saved: {rolling_ic_path}')

# %% [markdown]
# ## Section 4: Tone Distribution
#
# Load GDELT tone scores and compare across organic vs sponsored classifier groups.

# %%
# Load sponsored scores
SCORES_PATH = DATA_PROCESSED / 'sponsored_scores.parquet'
scores_df = pd.read_parquet(SCORES_PATH)

print(f'Loaded sponsored_scores: {len(scores_df):,} rows')
print(f'Columns: {list(scores_df.columns)}')

# Classify by sponsored_prob
high_organic = scores_df[scores_df['sponsored_prob'] < 0.3].copy()
uncertain = scores_df[(scores_df['sponsored_prob'] >= 0.3) & (scores_df['sponsored_prob'] <= 0.7)].copy()
high_sponsored = scores_df[scores_df['sponsored_prob'] > 0.7].copy()

high_organic['group'] = 'High Organic\n(sponsored_prob < 0.3)'
uncertain['group'] = 'Uncertain\n(0.3 ≤ prob ≤ 0.7)'
high_sponsored['group'] = 'High Sponsored\n(sponsored_prob > 0.7)'

tone_df = pd.concat([high_organic, uncertain, high_sponsored], ignore_index=True)

print(f'\nTone distribution by group:')
print(f'  High Organic: {len(high_organic):,} articles')
print(f'  Uncertain: {len(uncertain):,} articles')
print(f'  High Sponsored: {len(high_sponsored):,} articles')

# Compute mean tone for each group
tone_org = high_organic['tone_score'].dropna().mean()
tone_unc = uncertain['tone_score'].dropna().mean()
tone_sp = high_sponsored['tone_score'].dropna().mean()

print(f'\nMean GDELT TONE_SCORE:')
print(f'  High Organic: {tone_org:.4f}')
print(f'  Uncertain: {tone_unc:.4f}')
print(f'  High Sponsored: {tone_sp:.4f}')
print(f'  Difference (Organic - Sponsored): {tone_org - tone_sp:.4f}')

# Create violin plot
fig, ax = plt.subplots(figsize=(10, 7))

parts = ax.violinplot(
    [high_organic['tone_score'].dropna().values,
     uncertain['tone_score'].dropna().values,
     high_sponsored['tone_score'].dropna().values],
    positions=[0, 1, 2],
    showmeans=True,
    showmedians=False
)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['High Organic\n(sponsored_prob < 0.3)',
                    'Uncertain\n(0.3 ≤ prob ≤ 0.7)',
                    'High Sponsored\n(sponsored_prob > 0.7)'])
ax.set_ylabel('GDELT TONE_SCORE', fontsize=12)
ax.set_title('GDELT Tone Distribution by Article Classification', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
tone_fig_path = FIG_DIR_H1 / 'tone_violin_organic_vs_sponsored.png'
fig.savefig(tone_fig_path)
plt.close(fig)
print(f'\nSaved: {tone_fig_path}')

# %% [markdown]
# ## Section 5: Holdout Validation
#
# OUT-OF-SAMPLE: Repeat IC calculation on holdout period (date >= 2024-07-01) only.

# %%
# Filter to holdout period only
holdout_df = filtered_df[filtered_df['date'] >= '2024-07-01'].copy()
print(f'Holdout period (2024-07-01 onwards): {len(holdout_df):,} rows')

# Compute IC on holdout only
ic_org_holdout, _ = compute_daily_ic(holdout_df, 'organic_signal_norm', 'ret_close2close')
ic_sp_holdout, _ = compute_daily_ic(holdout_df, 'sponsored_signal_norm', 'ret_close2close')

print(f'Holdout IC days: Organic={len(ic_org_holdout)}, Sponsored={len(ic_sp_holdout)}')

# Statistics for holdout
if len(ic_org_holdout) > 1:
    ic_org_ho_mean = np.nanmean(ic_org_holdout)
    ic_org_ho_std = np.nanstd(ic_org_holdout, ddof=1)
    ic_org_ho_ir = ic_org_ho_mean / ic_org_ho_std if ic_org_ho_std > 0 else np.nan
    t_org_ho, p_org_ho = ttest_1samp(ic_org_holdout, 0, nan_policy='omit')
else:
    ic_org_ho_mean = np.nan
    ic_org_ho_std = np.nan
    ic_org_ho_ir = np.nan
    t_org_ho = np.nan
    p_org_ho = np.nan

if len(ic_sp_holdout) > 1:
    ic_sp_ho_mean = np.nanmean(ic_sp_holdout)
    ic_sp_ho_std = np.nanstd(ic_sp_holdout, ddof=1)
    ic_sp_ho_ir = ic_sp_ho_mean / ic_sp_ho_std if ic_sp_ho_std > 0 else np.nan
    t_sp_ho, p_sp_ho = ttest_1samp(ic_sp_holdout, 0, nan_policy='omit')
else:
    ic_sp_ho_mean = np.nan
    ic_sp_ho_std = np.nan
    ic_sp_ho_ir = np.nan
    t_sp_ho = np.nan
    p_sp_ho = np.nan

print('\n' + '='*80)
print('OUT-OF-SAMPLE HOLDOUT VALIDATION (2024-07-01+)')
print('='*80)
print(f'| {"Signal":<12} | {"N_days":>6} | {"IC_mean":>10} | {"ICIR":>8} | {"t_stat":>8} | {"p_value":>9} |')
print('|' + '-'*78 + '|')
print(f'| {"Organic":<12} | {len(ic_org_holdout):>6} | {ic_org_ho_mean:>10.6f} | {ic_org_ho_ir:>8.4f} | {t_org_ho:>8.4f} | {p_org_ho:>9.6f} |')
print(f'| {"Sponsored":<12} | {len(ic_sp_holdout):>6} | {ic_sp_ho_mean:>10.6f} | {ic_sp_ho_ir:>8.4f} | {t_sp_ho:>8.4f} | {p_sp_ho:>9.6f} |')
print('='*80)

# Store holdout results
results_holdout = {
    'ic_organic_mean': ic_org_ho_mean,
    'ic_organic_ir': ic_org_ho_ir,
    't_organic': t_org_ho,
    'p_organic': p_org_ho,
    'ic_sponsored_mean': ic_sp_ho_mean,
    'ic_sponsored_ir': ic_sp_ho_ir,
    't_sponsored': t_sp_ho,
    'p_sponsored': p_sp_ho,
}

# %% [markdown]
# ## Section 6: Interpretation

# %%
# Generate summary for interpretation
# Determine result: SUPPORTED if organic IC is significantly > sponsored IC
result_status = 'MIXED'
if results_train['p_diff'] < 0.05 and results_train['diff_mean'] > 0:
    result_status = 'SUPPORTED'
elif results_train['p_diff'] < 0.05 and results_train['diff_mean'] < 0:
    result_status = 'NOT SUPPORTED'

interpretation = f"""
### H1 Result: {result_status}

**Training Period (2020-2023):**
- Organic IC_mean: {results_train['ic_organic_mean']:.6f} (t={results_train['t_organic']:.4f}, p={results_train['p_organic']:.6f})
- Sponsored IC_mean: {results_train['ic_sponsored_mean']:.6f} (t={results_train['t_sponsored']:.4f}, p={results_train['p_sponsored']:.6f})
- IC Difference (Organic - Sponsored): {results_train['diff_mean']:.6f} (p={results_train['p_diff']:.6f})
  - Bootstrap 95% CI: [{results_train['ci_diff_low']:.6f}, {results_train['ci_diff_high']:.6f}]

**Holdout Period (2024-07-01+):**
- Organic IC_mean: {results_holdout['ic_organic_mean']:.6f}
- Sponsored IC_mean: {results_holdout['ic_sponsored_mean']:.6f}

**Interpretation:**
The analysis tests whether organic/editorial news carries stronger predictive signal than sponsored/PR content. 
The information coefficient measures daily cross-sectional correlation between normalized signals and next-day returns.
A positive organic IC suggests genuine predictive content, while a higher organic than sponsored IC would support H1.
The holdout validation assesses generalization to out-of-sample 2024 data.

**Caveats:**
- Sample size constraints during COVID (2020-03 to 2020-06) may affect IC reliability
- GDELT TONE_SCORE used as news sentiment proxy; FinBERT sentiment would be more robust
- Sponsored classifier trained on limited labeled data; misclassification may bias results
- Normalization based on train split (2020-2023) may not suit 2024 market regime
"""

print(interpretation)
