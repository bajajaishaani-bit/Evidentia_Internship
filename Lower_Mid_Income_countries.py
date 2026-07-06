"""
Stage 2: Simar-Wilson Double Bootstrap Truncated Regression
LOWER-MIDDLE AND LOW-INCOME COUNTRIES
World Bank definitions (2023):
- Lower-middle income: $1,136 - $4,465
- Low income: < $1,135

This version analyzes developing countries with lower GDP per capita
to compare against high-income results.

Method : Truncated regression MLE + vectorized parametric bootstrap
         following Simar & Wilson (2007) Algorithm 2
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import truncnorm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
FPE_PATH = '/Users/aishaanibajaj/PycharmProjects/Internship/FPE_scores.csv'
WGI_PATH = '/Users/aishaanibajaj/Downloads/Internship Evidentia /Second regression /WGI_cleaned.csv'
GINI_PATH = '/Users/aishaanibajaj/Downloads/Internship Evidentia /Second regression /GINI_cleaned.csv'
GDP_PATH = '/Users/aishaanibajaj/Downloads/GDP_perCapita_cleaned.csv'

YEAR_START = 2000
YEAR_END = 2021

# World Bank income thresholds (current USD, 2023)
HIGH_INCOME_THRESH = 13845
UPPER_MIDDLE_THRESH = 4465
LOWER_MIDDLE_THRESH = 1136
LOW_INCOME_THRESH = 1135

# Choose threshold for developing countries
# Option 1: Lower-middle + Low income (all developing)
GDP_THRESH_DEV = LOWER_MIDDLE_THRESH  # Keep countries with GDP < $4,465
# Option 2: Only low income
# GDP_THRESH_DEV = LOW_INCOME_THRESH   # Keep countries with GDP < $1,135
# Option 3: All except high-income
# GDP_THRESH_DEV = HIGH_INCOME_THRESH  # Keep countries with GDP < $13,845

N_BOOTSTRAP = 1000
MIN_OBS_PER_PARAM = 10

# FPE validation thresholds
FPE_MIN_THRESH = 0
FPE_MAX_THRESH = 1.0


# =============================================================================
# FPE SCORE VALIDATION
# =============================================================================

def validate_fpe_scores(fpe_data):
    """Validate FPE scores - treat 1.0 as valid frontier"""
    print("\n" + "=" * 65)
    print("  FPE SCORE VALIDATION CHECK")
    print("=" * 65)

    if 'fpe_score' not in fpe_data.columns:
        print("  WARNING: 'fpe_score' column not found in data!")
        return fpe_data

    n_initial = len(fpe_data)

    # Use tolerance for floating point comparisons
    valid_mask = (fpe_data['fpe_score'] >= FPE_MIN_THRESH) & (fpe_data['fpe_score'] <= FPE_MAX_THRESH + 1e-6)
    invalid_mask = ~valid_mask

    n_valid = valid_mask.sum()
    n_invalid = invalid_mask.sum()
    n_frontier = (np.abs(fpe_data['fpe_score'] - 1.0) < 1e-6).sum()

    print(f"\n  FPE Score Validation:")
    print(f"    Total observations         : {n_initial}")
    print(f"    Valid scores ([0, 1])      : {n_valid}")
    print(f"      - Frontier scores (=1)   : {n_frontier} (efficient DMUs)")
    print(f"      - Interior scores (<1)   : {n_valid - n_frontier}")
    print(f"    Invalid scores (outside 0-1): {n_invalid}")

    if n_invalid > 0:
        print(f"\n  ⚠️  Invalid scores found - checking if all are 1.0...")
        invalid_data = fpe_data[invalid_mask]
        if (invalid_data['fpe_score'] == 1.0).all():
            print("  ✅ All 'invalid' scores are exactly 1.0 (frontier)")
            print("  ✅ Treating as valid frontier observations")
            valid_mask = (fpe_data['fpe_score'] >= FPE_MIN_THRESH) & (fpe_data['fpe_score'] <= FPE_MAX_THRESH + 1e-6)
            n_valid = valid_mask.sum()
            n_invalid = (~valid_mask).sum()

    fpe_valid = fpe_data[valid_mask].copy()
    print(f"\n  ✅ Final valid FPE observations: {len(fpe_valid)}")
    print("=" * 65)

    return fpe_valid


# =============================================================================
# DATA LOADING
# =============================================================================

def melt_wide(filepath, value_name):
    """Load wide-format data and melt to long format"""
    df = pd.read_csv(filepath)

    if 'Year' in df.columns:
        df = df.rename(columns={'Year': 'year'})
        code_col = next((c for c in df.columns if c in ['iso3', 'Country Code', 'CountryCode']), None)
        if code_col and code_col != 'iso3':
            df = df.rename(columns={code_col: 'iso3'})
        val_col = next((c for c in df.columns if c not in ['iso3', 'year'] and c != code_col), None)
        if val_col and val_col != value_name:
            df = df.rename(columns={val_col: value_name})
        df = df[['iso3', 'year', value_name]]
        df['year'] = df['year'].astype(int)
        df[value_name] = pd.to_numeric(df[value_name], errors='coerce')
        df = df.dropna(subset=[value_name])
        return df

    code_col = next((c for c in df.columns if c in ['Country Code', 'iso3', 'CountryCode']), None)
    if code_col is None:
        raise ValueError(f"Cannot find country code column in {filepath}")
    df = df.rename(columns={code_col: 'iso3'})

    year_cols = [str(y) for y in range(YEAR_START, YEAR_END + 1) if str(y) in df.columns]
    if not year_cols:
        raise ValueError(f"No year columns found in {filepath}")

    long = df[['iso3'] + year_cols].melt(id_vars='iso3', var_name='year', value_name=value_name)
    long['year'] = long['year'].astype(int)
    long[value_name] = pd.to_numeric(long[value_name], errors='coerce')
    long = long.dropna(subset=[value_name])

    return long


def load_wgi_composite():
    """Load WGI and compute composite"""
    df = pd.read_csv(WGI_PATH)
    df = df.rename(columns={'Country Code': 'iso3'})
    dims = ['CC', 'GE', 'PV', 'RL', 'RQ', 'VA']

    records = []
    for yr in range(YEAR_START, YEAR_END + 1):
        cols = [f'WGI.{d}_{yr}' for d in dims]
        if not all(c in df.columns for c in cols):
            continue
        yr_df = df[['iso3'] + cols].copy()
        yr_df[cols] = yr_df[cols].apply(pd.to_numeric, errors='coerce')
        yr_df['wgi_composite'] = yr_df[cols].mean(axis=1)
        yr_df['year'] = yr
        records.append(yr_df[['iso3', 'year', 'wgi_composite']])

    wgi = pd.concat(records, ignore_index=True).dropna(subset=['wgi_composite'])
    return wgi


def build_stage2_panel():
    """Build full and developing country panels"""
    print("Loading data...")
    fpe = pd.read_csv(FPE_PATH)
    fpe = validate_fpe_scores(fpe)

    wgi = load_wgi_composite()
    gini = melt_wide(GINI_PATH, 'gini')
    gdp = melt_wide(GDP_PATH, 'gdp_pc')

    panel = (fpe.merge(wgi, on=['iso3', 'year'], how='inner')
             .merge(gini, on=['iso3', 'year'], how='left')
             .merge(gdp, on=['iso3', 'year'], how='left'))

    panel_full = panel.dropna(subset=['fpe_score', 'wgi_composite', 'gini']).copy()

    # Developing countries: lower-middle + low income
    panel_dev = panel_full[panel_full['gdp_pc'] < UPPER_MIDDLE_THRESH].copy()

    # Also create upper-middle income for comparison
    panel_upper_middle = panel_full[(panel_full['gdp_pc'] >= UPPER_MIDDLE_THRESH) &
                                    (panel_full['gdp_pc'] < HIGH_INCOME_THRESH)].copy()

    # Low income only
    panel_low = panel_full[panel_full['gdp_pc'] < LOW_INCOME_THRESH].copy()

    print(f"\nSample Sizes:")
    print(f"  Full sample          : {len(panel_full):>4} obs, {panel_full['iso3'].nunique():>2} countries")
    print(f"  Lower-middle + low   : {len(panel_dev):>4} obs, {panel_dev['iso3'].nunique():>2} countries")
    print(
        f"  Upper-middle only    : {len(panel_upper_middle):>4} obs, {panel_upper_middle['iso3'].nunique():>2} countries")
    print(f"  Low income only      : {len(panel_low):>4} obs, {panel_low['iso3'].nunique():>2} countries")

    return panel_full, panel_dev, panel_upper_middle, panel_low


# =============================================================================
# DESIGN MATRIX (POOLED PERIODS FOR SMALL SAMPLES)
# =============================================================================

def build_design_matrix_pooled(df):
    """Pool years into 5-year periods: const + wgi + gini + period_dummies"""
    d = df.copy()
    d['wgi_std'] = (d['wgi_composite'] - d['wgi_composite'].mean()) / d['wgi_composite'].std()
    d['gini_std'] = (d['gini'] - d['gini'].mean()) / d['gini'].std()

    # Create 5-year periods
    d['period'] = pd.cut(d['year'],
                         bins=[1999, 2004, 2009, 2014, 2021],
                         labels=['2000-04', '2005-09', '2010-14', '2015-21'])

    X = pd.DataFrame({
        'const': 1.0,
        'wgi': d['wgi_std'].values,
        'gini': d['gini_std'].values
    }, index=d.index)

    periods = sorted(d['period'].unique())
    if len(periods) > 1:
        for period in periods[1:]:
            X[f'period_{period}'] = (d['period'] == period).astype(float).values

    return X.values, X.columns.tolist()


def build_design_matrix_simple(df):
    """Simplified design matrix: const + wgi + gini + time_trend"""
    d = df.copy()
    d['wgi_std'] = (d['wgi_composite'] - d['wgi_composite'].mean()) / d['wgi_composite'].std()
    d['gini_std'] = (d['gini'] - d['gini'].mean()) / d['gini'].std()

    X = pd.DataFrame({
        'const': 1.0,
        'wgi': d['wgi_std'].values,
        'gini': d['gini_std'].values,
        'time_trend': (d['year'] - YEAR_START).values
    }, index=d.index)

    return X.values, X.columns.tolist()


# =============================================================================
# TRUNCATED REGRESSION MLE
# =============================================================================

def trunc_nll(params, y, X, upper=1.0):
    """Negative log-likelihood for truncated normal regression"""
    k = X.shape[1]
    beta = params[:k]
    sigma = np.exp(params[k])

    mu = X @ beta
    z_y = (y - mu) / sigma
    z_upper = (upper - mu) / sigma
    cdf_upper = np.clip(stats.norm.cdf(z_upper), 1e-10, 1.0)

    ll = stats.norm.logpdf(z_y) - np.log(sigma) - np.log(cdf_upper)
    return -np.sum(ll)


def fit_truncated_regression(y, X):
    """Fit truncated regression MLE"""
    k = X.shape[1]
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    logsig0 = np.log(max(np.std(y - X @ beta0), 1e-6))
    params0 = np.append(beta0, logsig0)

    res = minimize(trunc_nll, params0, args=(y, X),
                   method='L-BFGS-B',
                   options={'maxiter': 20000, 'ftol': 1e-14, 'gtol': 1e-8})

    return res.x[:k], np.exp(res.x[k]), res


# =============================================================================
# BOOTSTRAP
# =============================================================================

def parametric_bootstrap_adaptive(y, X, beta_hat, sigma_hat, n_bootstrap=None):
    """
    Bootstrap with adaptive settings for small samples
    """
    if n_bootstrap is None:
        n_bootstrap = min(N_BOOTSTRAP, 500 if len(y) < 50 else N_BOOTSTRAP)

    print(f"    Bootstrap draws: {n_bootstrap}")

    rng = np.random.default_rng(42)
    k, n = X.shape[1], len(y)
    mu_hat = X @ beta_hat

    a_vec = np.full(n, -10.0)
    b_vec = (1.0 - mu_hat) / sigma_hat

    boot_betas = []
    failed_draws = 0

    for b in range(n_bootstrap):
        seed_b = rng.integers(0, 2 ** 31)

        try:
            y_star = truncnorm.rvs(a_vec, b_vec,
                                   loc=mu_hat,
                                   scale=sigma_hat,
                                   random_state=int(seed_b))

            mask = y_star < 1.0
            if mask.sum() < k + 2:
                failed_draws += 1
                continue

            beta_b, _, res_b = fit_truncated_regression(y_star[mask], X[mask])
            if np.all(np.isfinite(beta_b)):
                boot_betas.append(beta_b)
            else:
                failed_draws += 1
        except Exception:
            failed_draws += 1

        if (b + 1) % 100 == 0:
            print(f"      Progress: {b + 1}/{n_bootstrap}, "
                  f"successful: {len(boot_betas)}, "
                  f"failed: {failed_draws}")

    boot_betas = np.array(boot_betas)
    success_rate = len(boot_betas) / n_bootstrap

    print(f"    Bootstrap complete: {len(boot_betas)}/{n_bootstrap} successful ({success_rate:.1%})")

    if success_rate < 0.5:
        print(f"    ⚠️  Low success rate - results may be unreliable")

    return boot_betas


# =============================================================================
# RUN STAGE 2
# =============================================================================

def run_stage2_developing(panel, label, design_method='pooled'):
    """
    Run Stage 2 specifically for developing country analysis
    """
    print(f"\n{'=' * 65}")
    print(f"  STAGE 2 — {label}")
    print(f"{'=' * 65}")

    if len(panel) == 0:
        print("  SKIPPED — empty panel.")
        return None

    # Remove frontier observations
    df_trunc = panel[panel['fpe_score'] < 0.9999].copy()
    n_frontier = len(panel) - len(df_trunc)

    print(f"  Total obs             : {len(panel)}")
    print(f"  Frontier removed (=1) : {n_frontier}")
    print(f"  Regression sample     : {len(df_trunc)}")

    # Build design matrix
    if design_method == 'pooled':
        X, col_names = build_design_matrix_pooled(df_trunc)
    else:
        X, col_names = build_design_matrix_simple(df_trunc)

    # Check sample adequacy
    n, k = X.shape
    n_per_param = n / k
    print(f"\n  Design Matrix:")
    print(f"    Observations: {n}")
    print(f"    Parameters: {k}")
    print(f"    n/k ratio: {n_per_param:.1f}")

    if n_per_param < MIN_OBS_PER_PARAM:
        print(f"    ⚠️  WARNING: n/k = {n_per_param:.1f} < {MIN_OBS_PER_PARAM}")
        print(f"    ⚠️  Results may be unreliable!")
    else:
        print(f"    ✅ Adequate sample size")

    print(f"\n  Regressors: {col_names}")
    print(f"  Fitting truncated regression MLE...")

    y = df_trunc['fpe_score'].values
    beta_hat, sigma_hat, res = fit_truncated_regression(y, X)

    status = "converged" if res.success else "did not fully converge"
    print(f"  Optimizer: {status}")
    print(f"  sigma: {sigma_hat:.4f}   log-L: {-res.fun:.2f}")

    # Bootstrap
    print(f"\n  Running bootstrap...")
    boot_betas = parametric_bootstrap_adaptive(y, X, beta_hat, sigma_hat)

    if len(boot_betas) == 0:
        print("  ❌ No successful bootstrap draws - cannot compute inference")
        return None

    # Inference
    se = boot_betas.std(axis=0)
    ci_low = np.percentile(boot_betas, 2.5, axis=0)
    ci_hi = np.percentile(boot_betas, 97.5, axis=0)
    t_stat = np.where(se > 0, beta_hat / se, np.nan)
    p_val = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

    # Results table
    print(f"\n  {'Variable':<15} {'Coef':>9} {'SE':>9} "
          f"{'t':>7} {'p':>8}  {'95% CI':<25} Sig")
    print(f"  {'-' * 75}")

    for i, name in enumerate(col_names):
        stars = ('***' if p_val[i] < 0.01 else
                 '**' if p_val[i] < 0.05 else
                 '*' if p_val[i] < 0.10 else '')
        ci_str = f"[{ci_low[i]:+.4f}, {ci_hi[i]:+.4f}]"
        print(f"  {name:<15} {beta_hat[i]:>+9.4f} {se[i]:>9.4f} "
              f"{t_stat[i]:>7.3f} {p_val[i]:>8.4f}  {ci_str:<25} {stars}")

    print(f"  {'-' * 75}")
    print(f"  sigma = {sigma_hat:.4f}  |  N = {len(df_trunc)} (excl. {n_frontier} frontier)")

    return {
        'label': label,
        'beta': beta_hat,
        'sigma': sigma_hat,
        'se': se,
        'ci_low': ci_low,
        'ci_high': ci_hi,
        't_stat': t_stat,
        'p_val': p_val,
        'col_names': col_names,
        'boot_betas': boot_betas,
        'n_total': len(panel),
        'n_trunc': len(df_trunc),
        'n_parameters': k,
        'n_per_param': n_per_param
    }


# =============================================================================
# COMPARISON AND SAVING
# =============================================================================

def compare_results(results_list):
    """Compare results across different samples"""
    print(f"\n{'=' * 65}")
    print("  COMPARISON ACROSS SAMPLES")
    print(f"{'=' * 65}")

    valid_results = [r for r in results_list if r is not None]
    if len(valid_results) < 2:
        print("  Not enough valid results for comparison")
        return

    print(f"\n  {'Sample':<25} {'n':>6} {'n/k':>6} {'WGI β':>10} {'WGI p':>8} "
          f"{'GINI β':>10} {'GINI p':>8}")
    print(f"  {'-' * 75}")

    for res in valid_results:
        wgi_idx = res['col_names'].index('wgi') if 'wgi' in res['col_names'] else None
        gini_idx = res['col_names'].index('gini') if 'gini' in res['col_names'] else None

        wgi_beta = res['beta'][wgi_idx] if wgi_idx is not None else np.nan
        wgi_p = res['p_val'][wgi_idx] if wgi_idx is not None else np.nan
        gini_beta = res['beta'][gini_idx] if gini_idx is not None else np.nan
        gini_p = res['p_val'][gini_idx] if gini_idx is not None else np.nan

        print(f"  {res['label']:<25} {res['n_trunc']:>6} {res['n_per_param']:>6.1f} "
              f"{wgi_beta:>+10.4f} {wgi_p:>8.4f} "
              f"{gini_beta:>+10.4f} {gini_p:>8.4f}")

    # Save results
    rows = []
    for res in valid_results:
        for i, name in enumerate(res['col_names']):
            rows.append({
                'sample': res['label'],
                'variable': name,
                'coef': round(res['beta'][i], 6),
                'se': round(res['se'][i], 6),
                't_stat': round(res['t_stat'][i], 4),
                'p_value': round(res['p_val'][i], 4),
                'ci_low': round(res['ci_low'][i], 6),
                'ci_high': round(res['ci_high'][i], 6),
                'n_obs': res['n_trunc'],
                'n_params': res['n_parameters'],
                'n_per_param': round(res['n_per_param'], 2)
            })

    pd.DataFrame(rows).to_csv('stage2_developing_results.csv', index=False)
    print(f"\n  Results saved to 'stage2_developing_results.csv'")


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def descriptive_stats(panel_full, panel_dev, panel_upper_middle, panel_low):
    """Generate descriptive statistics for all samples"""
    print(f"\n{'=' * 65}")
    print("  DESCRIPTIVE STATISTICS")
    print(f"{'=' * 65}")

    stats_list = []

    for name, panel in [('Full Sample', panel_full),
                        ('Lower-Middle + Low', panel_dev),
                        ('Upper-Middle Only', panel_upper_middle),
                        ('Low Income Only', panel_low)]:
        if len(panel) == 0:
            continue

        desc = {
            'Sample': name,
            'Observations': len(panel),
            'Countries': panel['iso3'].nunique(),
            'Years': f"{panel['year'].min()}-{panel['year'].max()}",
            'FPE Mean': f"{panel['fpe_score'].mean():.4f}",
            'FPE Std': f"{panel['fpe_score'].std():.4f}",
            'FPE Min': f"{panel['fpe_score'].min():.4f}",
            'FPE Max': f"{panel['fpe_score'].max():.4f}",
            'Frontier (%)': f"{(panel['fpe_score'] >= 0.9999).sum() / len(panel) * 100:.1f}%",
            'WGI Mean': f"{panel['wgi_composite'].mean():.3f}",
            'GINI Mean': f"{panel['gini'].mean():.1f}",
            'GDP Mean': f"${panel['gdp_pc'].mean():,.0f}"
        }
        stats_list.append(desc)

    df_stats = pd.DataFrame(stats_list)
    print("\n" + df_stats.to_string(index=False))

    df_stats.to_csv('stage2_developing_descriptive_stats.csv', index=False)
    print("\n  Descriptive statistics saved to 'stage2_developing_descriptive_stats.csv'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete developing countries analysis pipeline"""
    print("\n" + "=" * 65)
    print("  SIMAR-WILSON STAGE 2: DEVELOPING COUNTRIES ANALYSIS")
    print("  Lower-Middle and Low Income Countries")
    print("=" * 65)

    # Build panels
    panel_full, panel_dev, panel_upper_middle, panel_low = build_stage2_panel()

    # Descriptive statistics
    descriptive_stats(panel_full, panel_dev, panel_upper_middle, panel_low)

    # Run Stage 2 on different samples
    results = []

    # Full sample (for comparison)
    print(f"\n{'-' * 65}")
    print("  ANALYSIS 1: FULL SAMPLE (for comparison)")
    print(f"{'-' * 65}")
    res_full = run_stage2_developing(panel_full, 'Full Sample', design_method='pooled')
    results.append(res_full)

    # Lower-middle + Low income
    if len(panel_dev) > 30:
        print(f"\n{'-' * 65}")
        print("  ANALYSIS 2: LOWER-MIDDLE + LOW INCOME")
        print(f"{'-' * 65}")
        res_dev = run_stage2_developing(panel_dev, 'Lower-Middle + Low Income', design_method='pooled')
        results.append(res_dev)

    # Upper-middle only
    if len(panel_upper_middle) > 30:
        print(f"\n{'-' * 65}")
        print("  ANALYSIS 3: UPPER-MIDDLE INCOME ONLY")
        print(f"{'-' * 65}")
        res_upper = run_stage2_developing(panel_upper_middle, 'Upper-Middle Only', design_method='pooled')
        results.append(res_upper)

    # Low income only
    if len(panel_low) > 30:
        print(f"\n{'-' * 65}")
        print("  ANALYSIS 4: LOW INCOME ONLY")
        print(f"{'-' * 65}")
        res_low = run_stage2_developing(panel_low, 'Low Income Only', design_method='pooled')
        results.append(res_low)

    # Compare results
    compare_results(results)

    # Summary
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Samples analyzed:")
    print(f"    - Full sample: {len(panel_full)} obs, {panel_full['iso3'].nunique()} countries")
    print(f"    - Lower-middle + low: {len(panel_dev)} obs, {panel_dev['iso3'].nunique()} countries")
    print(f"    - Upper-middle: {len(panel_upper_middle)} obs, {panel_upper_middle['iso3'].nunique()} countries")
    print(f"    - Low income: {len(panel_low)} obs, {panel_low['iso3'].nunique()} countries")
    print("\n  See 'stage2_developing_results.csv' for detailed results")
    print("=" * 65)

    return results


if __name__ == '__main__':
    results = main()

    print("\n" + "=" * 65)
    print("  ANALYSIS COMPLETE")
    print("=" * 65)