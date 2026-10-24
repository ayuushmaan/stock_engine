"""Fast robustness test runner with holding period sweep."""
import json
import numpy as np
import pandas as pd
from scipy import stats

def main():
    df = pd.read_parquet('data/final/master_dataset.parquet')
    df = df.dropna(subset=['net_signal', 'ret_fwd_1d']).reset_index(drop=True)
    df = df[df['article_count_total'] >= 3].copy()
    date_col = 'date'
    return_col = 'ret_fwd_1d'
    
    # Pre-sort and get unique dates
    df = df.sort_values([date_col, 'ticker']).reset_index(drop=True)
    unique_dates = df[date_col].unique()
    
    signals = [
        ('net_signal', 'net'),
        ('organic_signal', 'organic'),
        ('sponsored_signal', 'sponsored'),
        ('signal_organic_closed', 'organic_closed'),
        ('signal_sponsored_closed', 'sponsored_closed'),
    ]
    
    all_results = {}
    all_p_values = []
    all_test_names = []
    
    for signal_col, label in signals:
        if signal_col not in df.columns:
            continue
        
        # Compute daily ICs using vectorized approach
        daily_ics = []
        for d in unique_dates:
            day_data = df[df[date_col] == d]
            valid = day_data[[signal_col, return_col]].notna().all(axis=1)
            if valid.sum() >= 5:
                s = day_data.loc[valid, signal_col].values
                r = day_data.loc[valid, return_col].values
                ic, _ = stats.spearmanr(s, r)
                daily_ics.append(ic)
        daily_ics = np.array(daily_ics)
        orig_mean_ic = float(np.nanmean(daily_ics))
        
        # Permutation test - 100 perms
        np.random.seed(42)
        placebo_means = []
        for _ in range(100):
            shuffled_returns = np.random.permutation(df[return_col].values)
            df_perm = df.copy()
            df_perm[return_col] = shuffled_returns
            
            perm_ics = []
            for d in unique_dates:
                day_data = df_perm[df_perm[date_col] == d]
                valid = day_data[[signal_col, return_col]].notna().all(axis=1)
                if valid.sum() >= 5:
                    s = day_data.loc[valid, signal_col].values
                    r = day_data.loc[valid, return_col].values
                    ic, _ = stats.spearmanr(s, r)
                    perm_ics.append(ic)
            perm_ics = np.array(perm_ics)
            if len(perm_ics) > 0:
                placebo_means.append(float(np.nanmean(perm_ics)))
        
        placebo_means = np.array(placebo_means)
        p_value = float(np.mean(placebo_means >= orig_mean_ic))
        
        result = {
            'original_mean_ic': round(orig_mean_ic, 6),
            'placebo_mean': round(float(np.nanmean(placebo_means)), 6),
            'placebo_std': round(float(np.nanstd(placebo_means)), 6),
            'p_value_twotailed': round(2*min(p_value,1-p_value), 6),
            'n_permutations': 100,
            'ci_95': [round(float(np.percentile(placebo_means,2.5)),6), round(float(np.percentile(placebo_means,97.5)),6)]
        }
        all_results[f'{label}_placebo'] = result
        all_p_values.append(result['p_value_twotailed'])
        all_test_names.append(f'{label}: placebo')
        print(f'{label}: IC={orig_mean_ic:.6f}, p={p_value:.4f}')
    
    # Global FDR
    n = len(all_p_values)
    if n > 0:
        sorted_indices = np.argsort(all_p_values)
        sorted_p = np.array(all_p_values)[sorted_indices]
        adjusted_p = np.zeros(n)
        for i in range(n): adjusted_p[i] = min(1.0, sorted_p[i] * n / (i+1))
        for i in range(n-2,-1,-1): adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
        res = np.zeros(n); res[sorted_indices] = adjusted_p
        global_fdr = res.tolist()
        for i, name in enumerate(all_test_names):
            key = name.replace(' ', '_')
            if key in all_results:
                all_results[key]['p_value_fdr_global'] = round(global_fdr[i], 6)
    
    # Holding period sweep
    print('\nRunning holding period sweep...')
    sweep_results = {}
    for h in range(1, 21):
        fwd_col = f'ret_fwd_{h}d'
        if fwd_col not in df.columns:
            break
        
        daily_records = []
        prev_longs = set()
        prev_shorts = set()
        
        for d in unique_dates:
            day_slice = df[df[date_col] == d]
            if len(day_slice) < 10:
                continue
            
            q_high = day_slice['net_signal'].quantile(0.80)
            q_low = day_slice['net_signal'].quantile(0.20)
            
            long_basket = day_slice[day_slice['net_signal'] >= q_high]
            short_basket = day_slice[day_slice['net_signal'] <= q_low]
            
            ret_long = float(long_basket[fwd_col].mean()) if len(long_basket) > 0 else 0.0
            ret_short = float(short_basket[fwd_col].mean()) if len(short_basket) > 0 else 0.0
            gross_return = ret_long - ret_short
            
            curr_longs = set(long_basket['ticker']) if 'ticker' in long_basket.columns else set()
            curr_shorts = set(short_basket['ticker']) if 'ticker' in short_basket.columns else set()
            
            if prev_longs and curr_longs:
                long_turnover = len(curr_longs - prev_longs) / max(1, len(curr_longs))
                short_turnover = len(curr_shorts - prev_shorts) / max(1, len(curr_shorts))
                turnover = (long_turnover + short_turnover) / 2.0
            else:
                turnover = 1.0
            
            prev_longs = curr_longs
            prev_shorts = curr_shorts
            
            tx_cost = 2.0 * turnover * (10.0 / 10000.0)
            net_return = gross_return - tx_cost
            
            daily_records.append({'date': d, 'gross': gross_return, 'net': net_return, 'turnover': turnover})
        
        if len(daily_records) == 0:
            continue
            
        perf_df = pd.DataFrame(daily_records)
        r_gross = perf_df['gross'].to_numpy()
        r_net = perf_df['net'].to_numpy()
        
        ann_gross = float(np.mean(r_gross) * 252.0)
        ann_net = float(np.mean(r_net) * 252.0)
        ann_vol = float(np.std(r_gross) * np.sqrt(252.0)) if len(r_gross) > 1 else 1e-6
        sharpe_gross = float(ann_gross / ann_vol) if ann_vol > 0 else 0.0
        sharpe_net = float(ann_net / ann_vol) if ann_vol > 0 else 0.0
        
        cum_net = np.cumprod(1.0 + r_net)
        peak = np.maximum.accumulate(cum_net)
        max_dd = float(np.min((cum_net - peak) / peak)) if len(cum_net) > 0 else 0.0
        avg_turnover = float(np.mean(perf_df['turnover']))
        
        sweep_results[f'h{h}'] = {
            'horizon_days': h,
            'n_days': len(perf_df),
            'ann_gross_return_pct': round(ann_gross * 100, 2),
            'ann_net_return_pct': round(ann_net * 100, 2),
            'ann_vol_pct': round(ann_vol * 100, 2),
            'sharpe_gross': round(sharpe_gross, 2),
            'sharpe_net': round(sharpe_net, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'avg_turnover_pct': round(avg_turnover * 100, 2),
            'cost_drag_pct': round(2 * avg_turnover * 10.0 / 100, 2),
        }
        print(f'  h{h}: net={sweep_results[f"h{h}"]["ann_net_return_pct"]:.1f}%, sharpe={sweep_results[f"h{h}"]["sharpe_net"]:.2f}, dd={sweep_results[f"h{h}"]["max_drawdown_pct"]:.1f}%, turnover={sweep_results[f"h{h}"]["avg_turnover_pct"]:.1f}%')
    
    all_results['holding_period_sweep'] = sweep_results
    
    # Save
    with open('outputs/tables/robustness_tests.json', 'w') as f: json.dump(all_results, f, indent=2, default=str)
    print('\nSaved robustness_tests.json')

if __name__ == '__main__':
    main()