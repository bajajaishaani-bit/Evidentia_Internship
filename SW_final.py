"""
Stage 2: Simar-Wilson Double Bootstrap Truncated Regression
Paper: Health Financing as Risk Transfer: Measuring Financial Protection
       Efficiency Across Countries

Method : Truncated regression MLE + vectorized parametric bootstrap
         following Simar & Wilson (2007) Algorithm 2
Regs   : WGI composite (avg 6 dims) + GINI + year fixed effects
Samples: Full + GDP-restricted (GDP per capita > $1000 current USD)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import truncnorm
from scipy.optimize import minimize

np.random.seed(42)

# =============================================================================
# CONFIG
# =============================================================================
FPE_PATH = '/Users/aishaanibajaj/PycharmProjects/Internship/FPE_scores.csv'
WGI_PATH = '/Users/aishaanibajaj/Downloads/Internship Evidentia /Second regression /WGI_cleaned.csv'
GINI_PATH = '/Users/aishaanibajaj/Downloads/Internship Evidentia /Second regression /GINI_cleaned.csv'
GDP_PATH = '/Users/aishaanibajaj/Downloads/GDP_perCapita_cleaned.csv'

YEAR_START = 2000
YEAR_END = 2021
GDP_THRESH = 13845  # current USD
N_BOOTSTRAP = 200

# FPE score validation thresholds
FPE_MIN_THRESH = 0  # Minimum acceptable FPE score
FPE_MAX_THRESH = 1.0  # Maximum acceptable FPE score (frontier = 1.0 is valid!)


# =============================================================================
# FPE SCORE CHECK FUNCTION (UPDATED - TREATS 1.0 AS VALID FRONTIER)
# =============================================================================

def validate_fpe_scores(fpe_data):
    """
    Validate FPE scores to ensure they are within acceptable bounds.
    NOTE: FPE = 1.0 is a VALID frontier score (efficient DMUs in DEA).
          Only scores outside [0, 1] are considered invalid.

    Parameters:
    -----------
    fpe_data : pandas.DataFrame
        DataFrame containing FPE scores (must have 'fpe_score' column)

    Returns:
    --------
    pandas.DataFrame : Validated FPE data with only valid scores
    """
    print("\n" + "=" * 65)
    print("  FPE SCORE VALIDATION CHECK")
    print("=" * 65)

    # Check if 'fpe_score' column exists
    if 'fpe_score' not in fpe_data.columns:
        print("  WARNING: 'fpe_score' column not found in data!")
        return fpe_data

    # Get initial statistics
    n_initial = len(fpe_data)

    # FPE scores are valid if they are in [0, 1]
    # NOTE: 1.0 is the frontier value (efficient DMU) - this is VALID
    valid_mask = (fpe_data['fpe_score'] >= FPE_MIN_THRESH) & (fpe_data['fpe_score'] <= FPE_MAX_THRESH)
    invalid_mask = ~valid_mask

    n_valid = valid_mask.sum()
    n_invalid = invalid_mask.sum()
    n_missing = fpe_data['fpe_score'].isna().sum()
    n_frontier = (fpe_data['fpe_score'] == 1.0).sum()

    # Report statistics
    print(f"\n  FPE Score Validation:")
    print(f"    Total observations         : {n_initial}")
    print(f"    Valid scores ([0, 1])      : {n_valid}")
    print(f"      - Frontier scores (=1)   : {n_frontier} (efficient DMUs)")
    print(f"      - Interior scores (<1)   : {n_valid - n_frontier}")
    print(f"    Invalid scores (outside 0-1): {n_invalid}")
    print(f"    Missing scores             : {n_missing}")

    # Report invalid scores if any
    if n_invalid > 0:
        invalid_data = fpe_data[invalid_mask]
        print(f"\n  ⚠️  Invalid FPE Scores Found:")
        print(f"    {len(invalid_data)} observations with FPE scores outside [0, 1] range")

        # Show summary of invalid values
        invalid_summary = invalid_data['fpe_score'].describe()
        print(f"\n    Invalid score summary:")
        print(f"      Min: {invalid_summary['min']:.6f}")
        print(f"      Max: {invalid_summary['max']:.6f}")
        print(f"      Mean: {invalid_summary['mean']:.6f}")
        print(f"      Std: {invalid_summary['std']:.6f}")

        # Show first few invalid rows
        print(f"\n    First {min(5, len(invalid_data))} invalid observations:")
        cols_to_show = ['iso3', 'year', 'fpe_score'] if all(c in invalid_data.columns for c in ['iso3', 'year']) else [
            'fpe_score']
        print(invalid_data[cols_to_show].head())

        print("\n  NOTE: Invalid FPE scores will be excluded from analysis")
    else:
        print("\n  ✅ All FPE scores are within valid range [0, 1]")

    # Report frontier statistics
    if n_frontier > 0:
        frontier_data = fpe_data[fpe_data['fpe_score'] == 1.0]
        print(f"\n  📊 Frontier Observations (FPE = 1.0):")
        print(f"    Total frontier DMUs: {n_frontier}")

        # Show distribution by year if available
        if 'year' in frontier_data.columns:
            year_counts = frontier_data['year'].value_counts().sort_index()
            print(f"    Distribution by year (top 5):")
            for year, count in year_counts.head(5).items():
                print(f"      {year}: {count} observations")
            if len(year_counts) > 5:
                print(f"      ... and {len(year_counts) - 5} more years")

    # Return ALL valid scores (including frontier = 1.0)
    fpe_valid = fpe_data[valid_mask].copy()

    print(f"\n  ✅ Final valid FPE observations: {len(fpe_valid)}")
    print(f"    (kept {n_frontier} frontier observations with FPE = 1.0)")
    if n_invalid > 0:
        print(f"    Filtered out {n_invalid} observations outside [0, 1]")
    print("=" * 65)

    return fpe_valid


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def melt_wide(filepath, value_name):
    """
    Load data - handles both wide format (year columns) and long format (Year column).
    """
    df = pd.read_csv(filepath)

    # Check if already in long format (has 'Year' column)
    if 'Year' in df.columns:
        # Already long format
        df = df.rename(columns={'Year': 'year'})
        # Find country code column
        code_col = next((c for c in df.columns
                         if c in ['iso3', 'Country Code', 'CountryCode']), None)
        if code_col and code_col != 'iso3':
            df = df.rename(columns={code_col: 'iso3'})
        # Keep only needed columns
        keep_cols = ['iso3', 'year', value_name]
        # Find the value column (might be GDP_per_capita, etc.)
        val_col = next((c for c in df.columns if c not in ['iso3', 'year'] and c != code_col), None)
        if val_col and val_col != value_name:
            df = df.rename(columns={val_col: value_name})
        df = df[['iso3', 'year', value_name]]
        df['year'] = df['year'].astype(int)
        df[value_name] = pd.to_numeric(df[value_name], errors='coerce')
        df = df.dropna(subset=[value_name])
        print(f"  {value_name:<15}: {df['iso3'].nunique():>3} countries, "
              f"{len(df):>5} obs  (years {df['year'].min()}-{df['year'].max()})")
        return df[['iso3', 'year', value_name]]

    # Otherwise, treat as wide format (original logic)
    code_col = next((c for c in df.columns
                     if c in ['Country Code', 'iso3', 'CountryCode']), None)
    if code_col is None:
        raise ValueError(f"Cannot find country code column in {filepath}")

    df = df.rename(columns={code_col: 'iso3'})

    year_cols = [str(y) for y in range(YEAR_START, YEAR_END + 1)
                 if str(y) in df.columns]

    if not year_cols:
        raise ValueError(f"No year columns {YEAR_START}-{YEAR_END} found in {filepath}. "
                         f"Columns present: {df.columns.tolist()[:10]}")

    long = df[['iso3'] + year_cols].melt(
        id_vars='iso3', var_name='year', value_name=value_name
    )
    long['year'] = long['year'].astype(int)
    long[value_name] = pd.to_numeric(long[value_name], errors='coerce')
    long = long.dropna(subset=[value_name])

    print(f"  {value_name:<15}: {long['iso3'].nunique():>3} countries, "
          f"{len(long):>5} obs  (years {long['year'].min()}-{long['year'].max()})")
    return long[['iso3', 'year', value_name]]


def load_wgi_composite():
    """
    Load WGI and compute composite as simple average of 6 governance dims
    per country-year. Columns: WGI.CC_YYYY, WGI.GE_YYYY, WGI.PV_YYYY,
                                WGI.RL_YYYY, WGI.RQ_YYYY, WGI.VA_YYYY
    Note: 2001 missing from WGI source data — those obs dropped at merge.
    """
    df = pd.read_csv(WGI_PATH)
    df = df.rename(columns={'Country Code': 'iso3'})
    dims = ['CC', 'GE', 'PV', 'RL', 'RQ', 'VA']

    records = []
    for yr in range(YEAR_START, YEAR_END + 1):
        cols = [f'WGI.{d}_{yr}' for d in dims]
        if not all(c in df.columns for c in cols):
            continue  # skip year if not all dims present (e.g. 2001)
        yr_df = df[['iso3'] + cols].copy()
        yr_df[cols] = yr_df[cols].apply(pd.to_numeric, errors='coerce')
        yr_df['wgi_composite'] = yr_df[cols].mean(axis=1)
        yr_df['year'] = yr
        records.append(yr_df[['iso3', 'year', 'wgi_composite']])

    wgi = pd.concat(records, ignore_index=True).dropna(subset=['wgi_composite'])
    print(f"  {'wgi_composite':<15}: {wgi['iso3'].nunique():>3} countries, "
          f"{len(wgi):>5} obs  (2001 absent — WGI source gap)")
    return wgi


# =============================================================================
# 2. BUILD STAGE 2 PANEL
# =============================================================================

def build_stage2_panel():
    print("Loading covariates:")
    fpe = pd.read_csv(FPE_PATH)

    # ===== FPE SCORE VALIDATION (UPDATED - TREATS 1.0 AS VALID FRONTIER) =====
    fpe = validate_fpe_scores(fpe)
    # ======================================================================

    wgi = load_wgi_composite()
    gini = melt_wide(GINI_PATH, 'gini')
    gdp = melt_wide(GDP_PATH, 'gdp_pc')

    panel = (fpe
             .merge(wgi, on=['iso3', 'year'], how='left')
             .merge(gini, on=['iso3', 'year'], how='left')
             .merge(gdp, on=['iso3', 'year'], how='left'))

    panel_full = panel.dropna(subset=['fpe_score', 'wgi_composite', 'gini']).copy()
    panel_restr = panel_full[panel_full['gdp_pc'] >= GDP_THRESH].copy()

    print(f"\nStage 2 panel:")
    print(f"  Full sample          : {len(panel_full):>4} obs, "
          f"{panel_full['iso3'].nunique()} countries")
    print(f"  Restricted (>=${GDP_THRESH:,}): {len(panel_restr):>4} obs, "
          f"{panel_restr['iso3'].nunique()} countries")

    if len(panel_restr) == 0:
        print("\n  WARNING: Restricted sample is empty.")
        print("  This likely means GDP_perCapita_cleaned.csv in your project")
        print("  folder is an old/different file. Please overwrite it with the")
        print("  file you uploaded — copy it from Downloads to your project dir.")

    return panel_full, panel_restr


# =============================================================================
# 3. TRUNCATED REGRESSION MLE
# =============================================================================

def build_design_matrix(df):
    """
    Design matrix: intercept + wgi (std) + gini (std) + year dummies.
    Standardizing continuous regressors for numerical stability.
    """
    d = df.copy()
    d['wgi_std'] = (d['wgi_composite'] - d['wgi_composite'].mean()) / d['wgi_composite'].std()
    d['gini_std'] = (d['gini'] - d['gini'].mean()) / d['gini'].std()

    X = pd.DataFrame({'const': 1.0,
                      'wgi': d['wgi_std'].values,
                      'gini': d['gini_std'].values}, index=d.index)

    # Year fixed effects (drop first year as baseline)
    for yr in sorted(d['year'].unique())[1:]:
        X[f'yr_{yr}'] = (d['year'] == yr).astype(float).values

    return X.values, X.columns.tolist()


def trunc_nll(params, y, X, upper=1.0):
    """
    Negative log-likelihood for right-truncated normal regression.
    Truncation at FPE = 1 (scores cannot exceed 1 by construction).
    """
    k = X.shape[1]
    beta = params[:k]
    sigma = np.exp(params[k])  # log-sigma for unconstrained optimization

    mu = X @ beta
    z_y = (y - mu) / sigma
    z_upper = (upper - mu) / sigma
    cdf_upper = np.clip(stats.norm.cdf(z_upper), 1e-10, 1.0)

    ll = stats.norm.logpdf(z_y) - np.log(sigma) - np.log(cdf_upper)
    return -np.sum(ll)


def fit_truncated_regression(y, X):
    """MLE for truncated regression. Returns beta_hat, sigma_hat, result."""
    k = X.shape[1]
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    logsig0 = np.log(max(np.std(y - X @ beta0), 1e-6))
    params0 = np.append(beta0, logsig0)

    res = minimize(trunc_nll, params0, args=(y, X),
                   method='L-BFGS-B',
                   options={'maxiter': 20000, 'ftol': 1e-14, 'gtol': 1e-8})

    return res.x[:k], np.exp(res.x[k]), res


# =============================================================================
# 4. VECTORIZED PARAMETRIC BOOTSTRAP
# =============================================================================

def parametric_bootstrap(y, X, beta_hat, sigma_hat, n_bootstrap=N_BOOTSTRAP):
    """
    Vectorized parametric bootstrap following Simar & Wilson (2007).

    For each bootstrap draw b:
      1. Draw y* from N(mu_hat_i, sigma^2) truncated above at 1 — VECTORIZED
      2. Keep y* < 1 (mirrors estimation step)
      3. Re-estimate truncated regression on y*
      4. Collect bootstrap beta distribution

    Vectorization replaces the per-observation inner loop with a single
    scipy.stats.truncnorm.rvs call, giving ~100x speedup.
    """
    rng = np.random.default_rng(42)
    k, n = X.shape[1], len(y)
    mu_hat = X @ beta_hat

    # Truncation bounds in standard normal units (upper bound per observation)
    a_vec = np.full(n, -10.0)  # effectively -inf
    b_vec = (1.0 - mu_hat) / sigma_hat  # upper bound per obs

    boot_betas = []

    for b in range(n_bootstrap):
        # Single vectorized draw from truncated normal
        seed_b = rng.integers(0, 2 ** 31)
        y_star = truncnorm.rvs(a_vec, b_vec,
                               loc=mu_hat,
                               scale=sigma_hat,
                               random_state=int(seed_b))

        # Keep only y_star < 1
        mask = y_star < 1.0
        if mask.sum() < k + 2:
            continue

        try:
            beta_b, _, res_b = fit_truncated_regression(y_star[mask], X[mask])
            if np.all(np.isfinite(beta_b)):
                boot_betas.append(beta_b)
        except Exception:
            pass

        # Progress indicator every 200 draws
        if (b + 1) % 200 == 0:
            print(f"    Bootstrap: {b + 1}/{n_bootstrap} "
                  f"({len(boot_betas)} successful)...")

    boot_betas = np.array(boot_betas)
    print(f"  Bootstrap complete: {len(boot_betas)}/{n_bootstrap} successful draws")
    return boot_betas


# =============================================================================
# 5. RUN STAGE 2
# =============================================================================

def run_stage2(panel, label):
    print(f"\n{'=' * 65}")
    print(f"  STAGE 2 — {label}")
    print(f"{'=' * 65}")

    if len(panel) == 0:
        print("  SKIPPED — empty panel.")
        return None

    # Remove observations with FPE exactly at the frontier (or extremely close)
    # This is standard for Simar-Wilson Algorithm 2 truncated regression
    df_trunc = panel[panel['fpe_score'] < 0.9999].copy()
    n_frontier = len(panel) - len(df_trunc)

    print(f"  Total obs             : {len(panel)}")
    print(f"  Frontier removed (=1) : {n_frontier}")
    print(f"  Regression sample     : {len(df_trunc)}")

    y, (X, col_names) = (df_trunc['fpe_score'].values,
                         build_design_matrix(df_trunc))

    print(f"\n  Regressors: {col_names[:3]} + {len(col_names) - 3} year FE")
    print(f"  Fitting truncated regression MLE...")

    beta_hat, sigma_hat, res = fit_truncated_regression(y, X)
    status = "converged" if res.success else "did not fully converge"
    print(f"  Optimizer  : {status}")
    print(f"  sigma      : {sigma_hat:.4f}   log-L : {-res.fun:.2f}")

    print(f"\n  Running bootstrap (B={N_BOOTSTRAP}) — vectorized, should be fast...")
    boot_betas = parametric_bootstrap(y, X, beta_hat, sigma_hat, N_BOOTSTRAP)

    # Inference
    se = boot_betas.std(axis=0)
    ci_low = np.percentile(boot_betas, 2.5, axis=0)
    ci_hi = np.percentile(boot_betas, 97.5, axis=0)
    t_stat = np.where(se > 0, beta_hat / se, np.nan)
    p_val = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

    # Print main results table
    print(f"\n  {'Variable':<10} {'Coef':>9} {'SE':>9} "
          f"{'t':>7} {'p':>8}  {'95% CI':<25} Sig")
    print(f"  {'-' * 75}")
    for i, name in enumerate(col_names[:3]):
        stars = ('***' if p_val[i] < 0.01 else
                 '**' if p_val[i] < 0.05 else
                 '*' if p_val[i] < 0.10 else '')
        ci_str = f"[{ci_low[i]:+.4f}, {ci_hi[i]:+.4f}]"
        print(f"  {name:<10} {beta_hat[i]:>+9.4f} {se[i]:>9.4f} "
              f"{t_stat[i]:>7.3f} {p_val[i]:>8.4f}  {ci_str:<25} {stars}")
    print(f"  {'-' * 75}")
    print(f"  sigma = {sigma_hat:.4f}  |  "
          f"N = {len(df_trunc)} (excl. {n_frontier} frontier obs)")

    return dict(label=label, beta=beta_hat, sigma=sigma_hat,
                se=se, ci_low=ci_low, ci_hi=ci_hi,
                t_stat=t_stat, p_val=p_val, col_names=col_names,
                boot_betas=boot_betas,
                n_total=len(panel), n_trunc=len(df_trunc))


# =============================================================================
# 6. COMPARISON + SAVE
# =============================================================================

def compare_and_save(res_full, res_restr):
    print(f"\n{'=' * 65}")
    print("  ROBUSTNESS: Full vs GDP-Restricted Sample")
    print(f"{'=' * 65}")

    if res_restr is None:
        print("  Restricted sample unavailable — fix GDP file and re-run.")
        return

    print(f"  {'Var':<8} {'Full β':>9} {'Full SE':>8} "
          f"{'Restr β':>9} {'Restr SE':>8}  Stable?")
    print(f"  {'-' * 55}")

    for var in ['const', 'wgi', 'gini']:
        if var in res_full['col_names'] and var in res_restr['col_names']:
            i_f = res_full['col_names'].index(var)
            i_r = res_restr['col_names'].index(var)
            b_f, se_f = res_full['beta'][i_f], res_full['se'][i_f]
            b_r, se_r = res_restr['beta'][i_r], res_restr['se'][i_r]

            same_sign = (b_f * b_r) > 0
            ci_overlap = (res_full['ci_low'][i_f] < res_restr['ci_hi'][i_r] and
                          res_restr['ci_low'][i_r] < res_full['ci_hi'][i_f])
            stable = 'YES ✓' if (same_sign and ci_overlap) else 'NO  ✗'
            print(f"  {var:<8} {b_f:>+9.4f} {se_f:>8.4f} "
                  f"{b_r:>+9.4f} {se_r:>8.4f}  {stable}")

    print(f"\n  Expected signs: wgi > 0 (governance helps)  "
          f"gini < 0 (inequality hurts)")

    # Save results CSV
    rows = []
    for res in [r for r in [res_full, res_restr] if r is not None]:
        for i, name in enumerate(res['col_names'][:3]):
            rows.append(dict(
                sample=res['label'], variable=name,
                coef=round(res['beta'][i], 6),
                se=round(res['se'][i], 6),
                t_stat=round(res['t_stat'][i], 4),
                p_value=round(res['p_val'][i], 4),
                ci_low=round(res['ci_low'][i], 6),
                ci_high=round(res['ci_hi'][i], 6),
            ))
    pd.DataFrame(rows).to_csv('stage2_results.csv', index=False)
    print(f"\n  Results saved to 'stage2_results.csv'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    panel_full, panel_restr = build_stage2_panel()

    res_full = run_stage2(panel_full, 'Full Sample')
    res_restr = run_stage2(panel_restr, f'GDP-Restricted (>=${GDP_THRESH})')

    compare_and_save(res_full, res_restr)