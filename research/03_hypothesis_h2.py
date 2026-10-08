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
# # H2: Closed-market vs Open-market Window News Signals
#
# **Hypothesis**: Closed-market-window news (3:30 PM to 9:15 AM IST) predicts returns more strongly than open-window news because it has not yet been priced in.

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
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

from config.settings import DATA_FINAL, OUTPUTS_FIGURES, seed_everything

seed_everything(42)

FIG_DIR_H2 = OUTPUTS_FIGURES / 'h2'
FIG_DIR_H2.mkdir(parents=True, exist_ok=True)

print(f"Project root: {ROOT}")
print(f"Figure output: {FIG_DIR_H2}")

# %% [markdown]
# ## Section 1: Signal-Return Alignment
#
# Alignment methodology:
# - `signal_organic_closed` on date t predicts `ret_overnight` on date t+1 (shifted by -1 day per ticker).
# - `signal_organic_open` on date t predicts `ret_intraday` on date t (same day).
# - `signal_sponsored_closed` on date t predicts `ret_overnight` on date t+1 (shifted by -1 day per ticker).
# - `signal_sponsored_open` on date t predicts `ret_intraday` on date t (same day).

# %%
# Load master dataset
master_df = pd.read_parquet(DATA_FINAL / 'master_dataset.parquet')
master_df['date'] = pd.to_datetime(master_df['date'])
master_df = master_df.sort_values(['date', 'ticker']).reset_index(drop=True)

# Filter for article_count_total >= 3
filtered_df = master_df[master_df['article_count_total'] >= 3].copy()
print(f"Loaded {len(master_df):,} rows, filtered to {len(filtered_df):,} rows (article_count_total >= 3)")

# Load Index Prices (IDX_NSEI) for Event Study abnormal return calculation
idx_df = pd.read_parquet(ROOT / 'data/raw/prices/IDX_NSEI.parquet').reset_index()
idx_df['date'] = pd.to_datetime(idx_df['date'])
idx_df = idx_df.sort_values('date').reset_index(drop=True)
idx_df['nifty_ret'] = idx_df['Close'].pct_change()
index_ret_dict = idx_df.set_index('date')['nifty_ret'].to_dict()

# Merge Index Returns into filtered_df
filtered_df = filtered_df.merge(idx_df[['date', 'nifty_ret']], on='date', how='left')
filtered_df['abnormal_return'] = filtered_df['ret_close2close'] - filtered_df['nifty_ret']

# Create four working series:
# df_oc: signal_organic_closed, ret_overnight shifted by -1 day per ticker
# df_oo: signal_organic_open,   ret_intraday same day
# df_sc: signal_sponsored_closed, ret_overnight shifted by -1 day per ticker
# df_so: signal_sponsored_open,   ret_intraday same day

df_oc = filtered_df.copy()
df_oc['ret_overnight_next'] = df_oc.groupby('ticker')['ret_overnight'].shift(-1)

# %% [markdown]
# ## Section 2: IC Test (all four signals)
#
# Compute daily cross-sectional IC (Spearman correlation) requiring >= 5 stocks.

# %%
def compute_daily_ic_series(df, signal_col, return_col):
    ic_list = []
    dates = []
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        valid_mask = day_data[[signal_col, return_col]].notna().all(axis=1)
        if valid_mask.sum() < 5:
            continue
        day_valid = day_data[valid_mask]
        signal = day_valid[signal_col].values
        returns = day_valid[return_col].values
        ic, _ = spearmanr(signal, returns)
        ic_list.append(ic)
        dates.append(date)
    return pd.DataFrame({'date': dates, 'ic': ic_list}).sort_values('date').reset_index(drop=True)

# Compute daily IC series
ic_df_oc = compute_daily_ic_series(df_oc, 'signal_organic_closed', 'ret_overnight_next')
ic_df_oo = compute_daily_ic_series(df_oc, 'signal_organic_open', 'ret_intraday')
ic_df_sc = compute_daily_ic_series(df_oc, 'signal_sponsored_closed', 'ret_overnight_next')
ic_df_so = compute_daily_ic_series(df_oc, 'signal_sponsored_open', 'ret_intraday')

def compute_ic_metrics(ic_df):
    ic_vals = ic_df['ic'].values
    mean_val = np.nanmean(ic_vals)
    std_val = np.nanstd(ic_vals, ddof=1)
    ir_val = mean_val / std_val if std_val > 0 else np.nan
    t_val, p_val = ttest_1samp(ic_vals, 0, nan_policy='omit')
    hit_rate = np.mean(ic_vals > 0) * 100
    rolling_ic = ic_df['ic'].rolling(window=60, min_periods=1).mean()
    stability = np.mean(rolling_ic > 0) * 100
    return {
        'N_days': len(ic_df),
        'IC_mean': mean_val,
        'IC_std': std_val,
        'ICIR': ir_val,
        't_stat': t_val,
        'p_value': p_val,
        'Hit_Rate': hit_rate,
        'Stability': stability
    }

oc_m = compute_ic_metrics(ic_df_oc)
oo_m = compute_ic_metrics(ic_df_oo)
sc_m = compute_ic_metrics(ic_df_sc)
so_m = compute_ic_metrics(ic_df_so)

print("| Signal                          | N_days | IC_mean | ICIR  | t_stat | p_value | Hit_Rate | Stability |")
print("|---------------------------------|--------|---------|-------|--------|---------|----------|-----------|")
print(f"| Organic  × Closed → overnight   | {oc_m['N_days']:>6} | {oc_m['IC_mean']:>7.6f} | {oc_m['ICIR']:>5.4f} | {oc_m['t_stat']:>6.4f} | {oc_m['p_value']:>7.6f} | {oc_m['Hit_Rate']:>7.2f}% | {oc_m['Stability']:>8.2f}% |")
print(f"| Organic  × Open   → intraday    | {oo_m['N_days']:>6} | {oo_m['IC_mean']:>7.6f} | {oo_m['ICIR']:>5.4f} | {oo_m['t_stat']:>6.4f} | {oo_m['p_value']:>7.6f} | {oo_m['Hit_Rate']:>7.2f}% | {oo_m['Stability']:>8.2f}% |")
print(f"| Sponsored × Closed → overnight  | {sc_m['N_days']:>6} | {sc_m['IC_mean']:>7.6f} | {sc_m['ICIR']:>5.4f} | {sc_m['t_stat']:>6.4f} | {sc_m['p_value']:>7.6f} | {sc_m['Hit_Rate']:>7.2f}% | {sc_m['Stability']:>8.2f}% |")
print(f"| Sponsored × Open   → intraday   | {so_m['N_days']:>6} | {so_m['IC_mean']:>7.6f} | {so_m['ICIR']:>5.4f} | {so_m['t_stat']:>6.4f} | {so_m['p_value']:>7.6f} | {so_m['Hit_Rate']:>7.2f}% | {so_m['Stability']:>8.2f}% |")

print("\n2x2 IC_mean grid:")
print("             | Closed | Open |")
print(f"  Organic    | {oc_m['IC_mean']:.6f} | {oo_m['IC_mean']:.6f} |")
print(f"  Sponsored  | {sc_m['IC_mean']:.6f} | {so_m['IC_mean']:.6f} |")

# Bootstrap
np.random.seed(42)
ic_oc = ic_df_oc['ic'].values
bootstrap_means = []
for _ in range(1000):
    idx = np.random.choice(len(ic_oc), size=len(ic_oc), replace=True)
    bootstrap_means.append(np.nanmean(ic_oc[idx]))
bootstrap_means = np.array(bootstrap_means)
ci_low = np.percentile(bootstrap_means, 2.5)
ci_high = np.percentile(bootstrap_means, 97.5)
print(f"\nBootstrap 95% CI on Organic×Closed IC_mean: [{ci_low:.6f}, {ci_high:.6f}]")

# %% [markdown]
# ## Section 3: Event Study
#
# CAR around High Organic Closed-Window Sentiment.

# %%
# Identify event thresholds on train split (date < '2024-01-01')
train_split = filtered_df[filtered_df['date'] < '2024-01-01'].copy()
ticker_p80 = train_split.groupby('ticker')['signal_organic_closed'].quantile(0.8)
ticker_p20 = train_split.groupby('ticker')['signal_organic_closed'].quantile(0.2)

bullish_events = []
bearish_events = []

for ticker, group in filtered_df.groupby('ticker'):
    p80 = ticker_p80.get(ticker, np.nan)
    p20 = ticker_p20.get(ticker, np.nan)
    if pd.isna(p80) or pd.isna(p20):
        continue
    group_sorted = group.sort_values('date').reset_index(drop=True)
    for i, row in group_sorted.iterrows():
        if row['article_count_total'] < 3:
            continue
        is_bull = row['signal_organic_closed'] >= p80
        is_bear = row['signal_organic_closed'] <= p20
        if is_bull or is_bear:
            if i - 2 >= 0 and i + 5 < len(group_sorted):
                window = group_sorted.iloc[i - 2 : i + 6]
                if not window['abnormal_return'].isna().any():
                    car = window['abnormal_return'].cumsum().values
                    if is_bull:
                        bullish_events.append(car)
                    if is_bear:
                        bearish_events.append(car)

bullish_events = np.array(bullish_events)
bearish_events = np.array(bearish_events)

n_bull = len(bullish_events)
n_bear = len(bearish_events)
print(f"n_bullish_events: {n_bull}")
print(f"n_bearish_events: {n_bear}")

k_values = np.array([-2, -1, 0, 1, 2, 3, 4, 5])

mean_car_bull = np.mean(bullish_events, axis=0)
stderr_car_bull = np.std(bullish_events, axis=0, ddof=1) / np.sqrt(n_bull)
t_stat_bull, p_val_bull = ttest_1samp(bullish_events, 0, axis=0)

mean_car_bear = np.mean(bearish_events, axis=0)
stderr_car_bear = np.std(bearish_events, axis=0, ddof=1) / np.sqrt(n_bear)

print(f"Bullish Event Day 0 CAR: {mean_car_bull[2]*100:.4f}% (t={t_stat_bull[2]:.4f})")
print(f"Bullish Event Day +1 CAR: {mean_car_bull[3]*100:.4f}% (t={t_stat_bull[3]:.4f})")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_car_bull * 100, color='#2563eb', linewidth=2, label=f'Bullish (n={n_bull})')
plt.fill_between(k_values, (mean_car_bull - 1.96 * stderr_car_bull) * 100, (mean_car_bull + 1.96 * stderr_car_bull) * 100, color='#2563eb', alpha=0.15)

plt.plot(k_values, mean_car_bear * 100, color='#dc2626', linewidth=2, linestyle='--', label=f'Bearish (n={n_bear})')
plt.fill_between(k_values, (mean_car_bear - 1.96 * stderr_car_bear) * 100, (mean_car_bear + 1.96 * stderr_car_bear) * 100, color='#dc2626', alpha=0.15)

plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)

plt.title("Event Study: CAR Around High Organic Closed-Window Sentiment", fontsize=14, fontweight='bold')
plt.xlabel("Trading Days Relative to Event Day (Day 0)", fontsize=12)
plt.ylabel("Cumulative Abnormal Return (CAR, %)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

event_plot_path = FIG_DIR_H2 / 'event_study_car.png'
plt.savefig(event_plot_path, dpi=200)
plt.close()
print(f"Saved: {event_plot_path}")

# %% [markdown]
# ## Section 4: Market Regime Breakdown
#
# Compute IC_mean breakdown across BULL, BEAR, and SIDEWAYS regimes.

# %%
date_regime = filtered_df.groupby('date')['market_regime'].first().to_dict()
ic_df_oc['regime'] = ic_df_oc['date'].map(date_regime)
ic_df_oo['regime'] = ic_df_oo['date'].map(date_regime)
ic_df_sc['regime'] = ic_df_sc['date'].map(date_regime)
ic_df_so['regime'] = ic_df_so['date'].map(date_regime)

regimes = ['BULL', 'BEAR', 'SIDEWAYS']
print("| Regime   | Org_Closed | Org_Open | Sp_Closed | Sp_Open |")
print("|----------|------------|----------|-----------|---------|")
for reg in regimes:
    moc = ic_df_oc[ic_df_oc['regime'] == reg]['ic'].mean()
    moo = ic_df_oo[ic_df_oo['regime'] == reg]['ic'].mean()
    msc = ic_df_sc[ic_df_sc['regime'] == reg]['ic'].mean()
    mso = ic_df_so[ic_df_so['regime'] == reg]['ic'].mean()
    print(f"| {reg:<8} | {moc:>10.6f} | {moo:>8.6f} | {msc:>9.6f} | {mso:>7.6f} |")

# %% [markdown]
# ## Section 5: Holdout Validation
#
# Repeat IC calculations on the Holdout period (date >= 2024-07-01).

# %%
ic_df_oc_ho = ic_df_oc[ic_df_oc['date'] >= '2024-07-01']
ic_df_oo_ho = ic_df_oo[ic_df_oo['date'] >= '2024-07-01']
ic_df_sc_ho = ic_df_sc[ic_df_sc['date'] >= '2024-07-01']
ic_df_so_ho = ic_df_so[ic_df_so['date'] >= '2024-07-01']

oc_ho = compute_ic_metrics(ic_df_oc_ho)
oo_ho = compute_ic_metrics(ic_df_oo_ho)
sc_ho = compute_ic_metrics(ic_df_sc_ho)
so_ho = compute_ic_metrics(ic_df_so_ho)

print("| Signal                          | N_days | IC_mean | ICIR  | t_stat | p_value | Hit_Rate | Stability |")
print("|---------------------------------|--------|---------|-------|--------|---------|----------|-----------|")
print(f"| Organic  × Closed → overnight   | {oc_ho['N_days']:>6} | {oc_ho['IC_mean']:>7.6f} | {oc_ho['ICIR']:>5.4f} | {oc_ho['t_stat']:>6.4f} | {oc_ho['p_value']:>7.6f} | {oc_ho['Hit_Rate']:>7.2f}% | {oc_ho['Stability']:>8.2f}% |")
print(f"| Organic  × Open   → intraday    | {oo_ho['N_days']:>6} | {oo_ho['IC_mean']:>7.6f} | {oo_ho['ICIR']:>5.4f} | {oo_ho['t_stat']:>6.4f} | {oo_ho['p_value']:>7.6f} | {oo_ho['Hit_Rate']:>7.2f}% | {oo_ho['Stability']:>8.2f}% |")
print(f"| Sponsored × Closed → overnight  | {sc_ho['N_days']:>6} | {sc_ho['IC_mean']:>7.6f} | {sc_ho['ICIR']:>5.4f} | {sc_ho['t_stat']:>6.4f} | {sc_ho['p_value']:>7.6f} | {sc_ho['Hit_Rate']:>7.2f}% | {sc_ho['Stability']:>8.2f}% |")
print(f"| Sponsored × Open   → intraday   | {so_ho['N_days']:>6} | {so_ho['IC_mean']:>7.6f} | {so_ho['ICIR']:>5.4f} | {so_ho['t_stat']:>6.4f} | {so_ho['p_value']:>7.6f} | {so_ho['Hit_Rate']:>7.2f}% | {so_ho['Stability']:>8.2f}% |")

# %%
# Paired difference test for interpretation
df_diff = ic_df_oc[['date', 'ic']].rename(columns={'ic': 'ic_oc'}).merge(
    ic_df_oo[['date', 'ic']].rename(columns={'ic': 'ic_oo'}),
    on='date',
    how='inner'
)
diff_s = df_diff['ic_oc'] - df_diff['ic_oo']
mean_diff = diff_s.mean()
t_diff, p_diff = ttest_1samp(diff_s, 0, nan_policy='omit')
print(f"Difference (Organic Closed - Organic Open): {mean_diff:.6f} (t={t_diff:.4f}, p={p_diff:.6f})")

# %% [markdown]
# ## Section 6: Interpretation
#
# H2 Result: NOT SUPPORTED
# Key finding: Sponsored × Open → intraday has the highest IC_mean (0.040965).
# Organic Closed IC_mean: -0.016 (t=-2.82, p=0.005)
# Organic Open IC_mean: 0.018 (t=2.92, p=0.004)
# Difference: -0.034 (p=0.000)
# Event CAR at day+1: 0.262% (t=7.03)
# Holdout organic closed IC: -0.016
#
# Interpretation: Closed-window news sentiment does not predict returns more strongly than open-window news; in fact, closed-window sentiment has a negative correlation with overnight returns. Intraday signals are consistently positive and statistically significant. This indicates that overnight news triggers immediate overnight price reversals or noise-driven reactions, whereas intraday sentiment alignment is more persistent.
