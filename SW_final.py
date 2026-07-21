"""
================================================================================
STAGE 2: SIMAR-WILSON DOUBLE-BOOTSTRAP TRUNCATED REGRESSION
================================================================================
REBUILT VERSION
- Fixed p-value calculation (was testing beta_corrected against its own
  bootstrap distribution instead of testing against zero)
- Fixed FD_Overall / FD component models to exclude parent/mid-level FD
  indices (Financial_Development_Index, Financial_Institutions_Index,
  Financial_Markets_Index) from any model that also includes their own
  sub-components, to avoid regressing an aggregate on pieces of itself
- Removed duplicated / conflicting composite-construction logic
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# EDIT THESE FILE PATHS - Change these to match your file names and locations
# ============================================================================

STAGE1_FILE = '/Users/aishaanibajaj/PycharmProjects/Internship/efficiency_scores_for_stage2.csv'
WGI_FILE = '/Users/aishaanibajaj/Downloads/wgi_country_means.csv'
FD_FILE = '/Users/aishaanibajaj/Downloads/FDI WHO.csv'

OUTPUT_PREFIX = 'simar_wilson_results'

BOOTSTRAP_ITERATIONS = 500          # bump to 2000 for final results
SIGNIFICANCE_LEVEL = 0.05

# Parent / mid-level FD indices that are built FROM the leaf components below.
# These must never appear in the same model as their own children.
FD_PARENT_INDICES = [
    'Financial_Development_Index',
    'Financial_Institutions_Index',
    'Financial_Markets_Index'
]

# ============================================================================
# PART 1: TRUNCATED REGRESSION CLASS
# ============================================================================

class TruncatedRegression:
    """Truncated regression with lower bound 0 and upper bound 1"""

    def __init__(self, lower_bound=0, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.params = None
        self.sigma = None

    def log_likelihood(self, params, X, y):
        beta = params[:-1]
        sigma = params[-1]

        if sigma <= 0:
            return 1e10

        mu = X @ beta
        z_lower = (self.lower_bound - mu) / sigma
        z_upper = (self.upper_bound - mu) / sigma

        ll = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2
        ll -= np.log(stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower) + 1e-10)

        return -np.sum(ll)

    def fit(self, X, y, max_iter=1000):
        try:
            beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            beta_init = np.zeros(X.shape[1])

        sigma_init = np.std(y - X @ beta_init) if len(y) > 1 else 0.1
        if sigma_init < 0.01:
            sigma_init = 0.1

        params_init = np.append(beta_init, sigma_init)

        result = minimize(
            self.log_likelihood,
            params_init,
            args=(X, y),
            method='L-BFGS-B',
            bounds=[(None, None)] * len(beta_init) + [(1e-10, None)],
            options={'maxiter': max_iter}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")

        self.params = result.x[:-1]
        self.sigma = result.x[-1]

        return self

    def predict(self, X):
        return X @ self.params


# ============================================================================
# PART 2: SIMAR-WILSON BOOTSTRAP
# ============================================================================

def simar_wilson_bootstrap(X, y, B1=500, alpha=0.05):
    """Simar-Wilson double-bootstrap for truncated regression"""
    n_obs, n_vars = X.shape

    model = TruncatedRegression(lower_bound=0, upper_bound=1)
    model.fit(X, y)
    beta_initial = model.params
    sigma_initial = model.sigma

    print(f"  Initial Sigma: {sigma_initial:.6f}")

    beta_boot = np.zeros((B1, n_vars))

    print(f"\n  Running {B1} bootstrap iterations...")

    for i in range(B1):
        if (i + 1) % 100 == 0:
            print(f"    Iteration {i + 1}/{B1}...")

        mu = X @ beta_initial

        u_uniform = np.random.uniform(0, 1, n_obs)
        z_lower = (0 - mu) / sigma_initial
        z_upper = (1 - mu) / sigma_initial

        phi_lower = stats.norm.cdf(z_lower)
        phi_upper = stats.norm.cdf(z_upper)

        u_trunc = phi_lower + u_uniform * (phi_upper - phi_lower)
        u_trunc = np.clip(u_trunc, 1e-10, 1 - 1e-10)
        epsilon_boot = sigma_initial * stats.norm.ppf(u_trunc)

        y_boot = mu + epsilon_boot
        y_boot = np.clip(y_boot, 0.001, 0.999)

        boot_model = TruncatedRegression(lower_bound=0, upper_bound=1)
        try:
            boot_model.fit(X, y_boot, max_iter=500)
            beta_boot[i, :] = boot_model.params
        except Exception:
            beta_boot[i, :] = beta_initial

    print("  Bootstrap complete!\n")

    beta_boot_mean = np.mean(beta_boot, axis=0)
    beta_boot_std = np.std(beta_boot, axis=0)

    # Bias correction
    bias = beta_boot_mean - beta_initial
    beta_corrected = beta_initial - bias

    # Bias-corrected percentile confidence intervals.
    # Reflected around beta_initial rather than raw percentiles of beta_boot,
    # consistent with the bias correction applied to the point estimate.
    raw_lower = np.percentile(beta_boot, alpha / 2 * 100, axis=0)
    raw_upper = np.percentile(beta_boot, (1 - alpha / 2) * 100, axis=0)
    ci_lower = 2 * beta_initial - raw_upper
    ci_upper = 2 * beta_initial - raw_lower

    # P-values: fraction of the bootstrap distribution that crosses zero,
    # i.e. testing against the null of no effect (NOT against beta_corrected
    # itself, which would just test where the estimate sits within its own
    # sampling distribution and trivially return ~0.5 for everything).
    p_values = np.zeros(n_vars)
    for j in range(n_vars):
        p_lower = np.mean(beta_boot[:, j] <= 0)
        p_upper = np.mean(beta_boot[:, j] >= 0)
        p_values[j] = 2 * min(p_lower, p_upper)
        p_values[j] = np.clip(p_values[j], 0, 1)

    return {
        'beta_initial': beta_initial,
        'beta_corrected': beta_corrected,
        'beta_boot_std': beta_boot_std,
        'bias': bias,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_values': p_values,
        'sigma_initial': sigma_initial,
        'beta_boot': beta_boot
    }


# ============================================================================
# PART 3: DATA LOADING FUNCTIONS
# ============================================================================

def load_wgi_data(file_path):
    df = pd.read_csv(file_path)
    print(f"  WGI columns: {list(df.columns)}")
    return df


def load_fd_data(file_path):
    """Load and prepare FD data (wide format with years as columns)"""
    df = pd.read_csv(file_path)
    print(f"  FD columns: {list(df.columns)}")

    year_cols = [col for col in df.columns if col.isdigit() and int(col) >= 1980]

    if not year_cols:
        print("  No year columns found (columns like 1980, 1981, etc.)")
        return df

    print(f"  Found year columns: {year_cols[0]} to {year_cols[-1]}")

    id_vars = ['COUNTRY', 'INDICATOR', 'SERIES_CODE', 'SERIES_NAME']
    id_vars = [col for col in id_vars if col in df.columns]

    df_long = pd.melt(df, id_vars=id_vars, value_vars=year_cols,
                       var_name='Year', value_name='FD_Value')

    df_long['Year'] = df_long['Year'].astype(int)
    df_long['FD_Value'] = pd.to_numeric(df_long['FD_Value'], errors='coerce')
    df_long = df_long.dropna(subset=['FD_Value'])

    if 'INDICATOR' not in df_long.columns:
        print("  No INDICATOR column found, returning long format")
        return df_long

    df_pivot = df_long.pivot_table(
        index=['COUNTRY', 'Year'], columns='INDICATOR', values='FD_Value'
    ).reset_index()

    df_pivot.columns = [
        col.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        for col in df_pivot.columns
    ]
    df_pivot.rename(columns={'COUNTRY': 'Country Name'}, inplace=True)

    print(f"  Created FD data with {len(df_pivot)} observations")
    print(f"  FD indicators: {[c for c in df_pivot.columns if c not in ['Country Name', 'Year']]}")
    return df_pivot


# ============================================================================
# PART 4: MAIN ANALYSIS
# ============================================================================

print("=" * 70)
print("STAGE 2: SIMAR-WILSON DOUBLE-BOOTSTRAP TRUNCATED REGRESSION")
print("=" * 70)
print("\nFile paths being used:")
print(f"  Stage 1 DEA results: {STAGE1_FILE}")
print(f"  WGI data: {WGI_FILE}")
print(f"  FD data: {FD_FILE}")
print(f"  Output prefix: {OUTPUT_PREFIX}")
print(f"  Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")

# ----------------------------------------------------------------------------
# Step 1: Load Stage 1 Results
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 1: LOADING STAGE 1 RESULTS")
print("-" * 70)

if not os.path.exists(STAGE1_FILE):
    print(f"\n❌ File not found: {STAGE1_FILE}")
    exit()

df_stage1 = pd.read_csv(STAGE1_FILE)
print(f"✓ Loaded: {STAGE1_FILE}")
print(f"  Observations: {len(df_stage1)}")

required_stage1 = ['DEA_Efficiency', 'Year']
missing = [c for c in required_stage1 if c not in df_stage1.columns]
if missing:
    print(f"\n❌ Missing columns in stage 1 file: {missing}")
    exit()

print(f"  Years: {df_stage1['Year'].min()} to {df_stage1['Year'].max()}")
print(f"  Countries: {df_stage1['Country Name'].nunique()} unique countries")

# ----------------------------------------------------------------------------
# Step 2: Load WGI Data
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 2: LOADING WGI DATA")
print("-" * 70)

if not os.path.exists(WGI_FILE):
    print(f"\n❌ File not found: {WGI_FILE}")
    exit()

df_wgi = load_wgi_data(WGI_FILE)
print(f"✓ Loaded: {WGI_FILE}")
print(f"  Observations: {len(df_wgi)}")
print(f"  Countries: {df_wgi['Country Name'].nunique()} unique countries")

# ----------------------------------------------------------------------------
# Step 3: Load Financial Development Data
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 3: LOADING FINANCIAL DEVELOPMENT DATA")
print("-" * 70)

if not os.path.exists(FD_FILE):
    print(f"\n❌ File not found: {FD_FILE}")
    exit()

df_fd = load_fd_data(FD_FILE)
print(f"✓ Loaded and processed FD data")
print(f"  Observations: {len(df_fd)}")

# ----------------------------------------------------------------------------
# Step 4: Merge All Data
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 4: MERGING ALL DATA")
print("-" * 70)

if 'Country Name' not in df_fd.columns:
    print("  Renaming COUNTRY to Country Name for consistency...")
    df_fd.rename(columns={'COUNTRY': 'Country Name'}, inplace=True)

df_merged = pd.merge(df_stage1, df_wgi, on=['Country Name'], how='inner')
print(f"  After WGI merge: {len(df_merged)} observations")

if len(df_merged) == 0:
    print("\n❌ No matches found with WGI data!")
    exit()

df_merged = pd.merge(df_merged, df_fd, on=['Country Name', 'Year'], how='inner')
print(f"  After FD merge: {len(df_merged)} observations")

if len(df_merged) == 0:
    print("\n❌ No matches found with FD data!")
    exit()

print(f"  Final merged data: {len(df_merged)} observations")
print(f"  Years: {df_merged['Year'].min()} to {df_merged['Year'].max()}")
print(f"  Countries: {df_merged['Country Name'].nunique()} unique countries")

# ----------------------------------------------------------------------------
# Step 5: Identify WGI and FD Columns
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 5: IDENTIFYING WGI AND FD COLUMNS")
print("-" * 70)

all_cols = list(df_merged.columns)

# WGI columns
wgi_columns = [col for col in all_cols if 'WGI' in col]

# FD columns: every numeric column that isn't an ID/WGI/outcome column.
# This includes parent indices AND their leaf components - we separate
# those out below, in Step 6, rather than here.
exclude_cols = ['Country Name', 'Year', 'DEA_Efficiency', 'FPE', 'CHE_PPP',
                 'GGHE_pct', 'Country Code_x', 'Country Code_y']

fd_columns = []
for col in all_cols:
    if col not in exclude_cols + wgi_columns:
        try:
            pd.to_numeric(df_merged[col], errors='raise')
            fd_columns.append(col)
        except Exception:
            print(f"  Skipping non-numeric column: {col}")

print(f"  Identified WGI columns: {wgi_columns}")
print(f"  Identified FD columns (parents + leaves): {fd_columns}")

# ----------------------------------------------------------------------------
# Step 6: Create Composite Indicators
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 6: CREATING COMPOSITE INDICATORS")
print("-" * 70)

# --- WGI Composite ---
if 'WGI_Composite' in df_merged.columns:
    print("✓ WGI_Composite already exists")
else:
    available_wgi = [c for c in wgi_columns if c != 'WGI_composite']
    if available_wgi:
        df_merged['WGI_Composite'] = df_merged[available_wgi].mean(axis=1)
        print(f"✓ WGI_Composite created from {len(available_wgi)} components: {available_wgi}")
    else:
        print("⚠️  No WGI components found to build WGI_Composite")

# --- FD: separate parent/mid-level indices from leaf components ---
available_fd = [c for c in fd_columns if c in df_merged.columns]
available_fd_leaves = [c for c in available_fd if c not in FD_PARENT_INDICES]

if not available_fd_leaves:
    print("⚠️  No leaf-level FD columns found")
else:
    for col in available_fd_leaves:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    # FD_Overall built ONLY from leaf-level components (Access/Depth/Efficiency
    # x Institutions/Markets) - excludes Financial_Development_Index and the
    # two mid-level indices, since those are themselves aggregates of these
    # same leaves and including them would double-count.
    df_merged['FD_Overall'] = df_merged[available_fd_leaves].mean(axis=1)
    print(f"✓ FD_Overall created from {len(available_fd_leaves)} leaf-level components:")
    print(f"  {available_fd_leaves}")

excluded_parents_present = [c for c in FD_PARENT_INDICES if c in available_fd]
if excluded_parents_present:
    print(f"  (Excluded parent/mid-level indices from FD_Overall: {excluded_parents_present})")

# ----------------------------------------------------------------------------
# Step 7: Clean Data
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 7: CLEANING DATA")
print("-" * 70)

keep_cols = ['DEA_Efficiency', 'Year', 'WGI_Composite', 'FD_Overall']
keep_cols.extend([c for c in wgi_columns if c in df_merged.columns and c != 'WGI_composite'])
keep_cols.extend([c for c in available_fd if c in df_merged.columns])
keep_cols = list(set(keep_cols))

for col in ['DEA_Efficiency', 'Year', 'WGI_Composite', 'FD_Overall']:
    if col not in df_merged.columns:
        print(f"⚠️  {col} not found in data!")
        exit()

df_clean = df_merged[keep_cols].copy()

for col in df_clean.columns:
    if col not in ['Country Name', 'Year']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = df_clean.dropna()
print(f"  Observations after cleaning: {len(df_clean)}")

if len(df_clean) < 10:
    print(f"\n⚠️  Very few observations ({len(df_clean)}) after cleaning.")
    for col in df_clean.columns:
        print(f"    {col}: {df_clean[col].notna().sum()}")
    exit()

year_dummies = pd.get_dummies(df_clean['Year'], prefix='Year', drop_first=True)
print(f"  Year dummies created: {len(year_dummies.columns)} years")

# ----------------------------------------------------------------------------
# Step 8: Define Models
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 8: MODEL SPECIFICATIONS")
print("-" * 70)

available_wgi = [c for c in wgi_columns if c in df_clean.columns and c != 'WGI_composite']
available_fd_leaves_clean = [c for c in available_fd_leaves if c in df_clean.columns]

models = {
    'Model_1_Aggregated': ['WGI_Composite', 'FD_Overall'],
    'Model_2_WGI_Components': available_wgi + ['FD_Overall'],
    # Model 3 and 4 use leaf-level FD components only - no parent/mid-level
    # indices - to avoid regressing an aggregate on its own pieces.
    'Model_3_FD_Components': ['WGI_Composite'] + available_fd_leaves_clean,
    'Model_4_Fully_Disaggregated': available_wgi + available_fd_leaves_clean
}

models = {k: v for k, v in models.items() if len(v) > 0}

print(f"  Running {len(models)} models:")
for name, vars_list in models.items():
    available = [v for v in vars_list if v in df_clean.columns]
    print(f"    - {name}: {len(available)} variables -> {available}")
    missing_v = [v for v in vars_list if v not in df_clean.columns]
    if missing_v:
        print(f"      Missing: {missing_v}")

# ----------------------------------------------------------------------------
# Step 9: Run Simar-Wilson for Each Model
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 9: RUNNING SIMAR-WILSON BOOTSTRAP")
print("-" * 70)

all_results = {}

for model_name, vars_list in models.items():
    print(f"\n{'=' * 70}")
    print(f"Running: {model_name}")
    print(f"{'=' * 70}")

    available_vars = [v for v in vars_list if v in df_clean.columns]
    if not available_vars:
        print("  No variables available, skipping...")
        continue

    year_cols = list(year_dummies.columns)
    X_cols = ['const'] + available_vars + year_cols

    X = np.column_stack(
        [np.ones(len(df_clean))]
        + [df_clean[v].values for v in available_vars]
        + [year_dummies[c].values for c in year_cols]
    )
    y = df_clean['DEA_Efficiency'].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.5, posinf=0.5, neginf=0.5)

    print(f"  Observations: {len(y)}")
    print(f"  Variable names: {X_cols}")

    results = simar_wilson_bootstrap(X, y, B1=BOOTSTRAP_ITERATIONS, alpha=SIGNIFICANCE_LEVEL)
    results['X_cols'] = X_cols
    results['model_name'] = model_name
    all_results[model_name] = results

# ----------------------------------------------------------------------------
# Step 10: Display Results
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: RESULTS")
print("=" * 70)

for model_name, results in all_results.items():
    print(f"\n{'=' * 70}")
    print(f"{model_name}")
    print(f"{'=' * 70}")
    print(f"  Bootstrap Iterations: {BOOTSTRAP_ITERATIONS}")
    print(f"  Sigma: {results['sigma_initial']:.6f}")
    print(f"\n  {'Variable':<35} {'Coefficient':>12} {'Std Error':>12} {'Bias':>12} "
          f"{f'{SIGNIFICANCE_LEVEL / 2 * 100:.1f}% CI':>14} "
          f"{f'{(1 - SIGNIFICANCE_LEVEL / 2) * 100:.1f}% CI':>14} {'P-value':>10}")
    print(f"  {'-' * 130}")

    for i, var_name in enumerate(results['X_cols']):
        p_val = results['p_values'][i]
        p_str = f"{p_val:.4f}" if p_val > 0.0001 else "<0.0001"

        stars = ""
        if p_val < 0.01:
            stars = "***"
        elif p_val < 0.05:
            stars = "**"
        elif p_val < 0.10:
            stars = "*"

        print(f"  {var_name:<35} {results['beta_corrected'][i]:>12.6f} "
              f"{results['beta_boot_std'][i]:>12.6f} "
              f"{results['bias'][i]:>12.6f} "
              f"{results['ci_lower'][i]:>14.6f} "
              f"{results['ci_upper'][i]:>14.6f} "
              f"{p_str:>10} {stars}")

# ----------------------------------------------------------------------------
# Step 11: Save Results
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 11: SAVING RESULTS")
print("-" * 70)

for model_name, results in all_results.items():
    results_df = pd.DataFrame({
        'Variable': results['X_cols'],
        'Coefficient': results['beta_corrected'],
        'Std_Error': results['beta_boot_std'],
        'Bias': results['bias'],
        'CI_Lower': results['ci_lower'],
        'CI_Upper': results['ci_upper'],
        'P_Value': results['p_values']
    })
    filename = f'{OUTPUT_PREFIX}_{model_name}.csv'
    results_df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")

all_results_df = pd.DataFrame()
for model_name, results in all_results.items():
    temp_df = pd.DataFrame({
        'Model': model_name,
        'Variable': results['X_cols'],
        'Coefficient': results['beta_corrected'],
        'Std_Error': results['beta_boot_std'],
        'Bias': results['bias'],
        'CI_Lower': results['ci_lower'],
        'CI_Upper': results['ci_upper'],
        'P_Value': results['p_values']
    })
    all_results_df = pd.concat([all_results_df, temp_df])

all_results_df.to_csv(f'{OUTPUT_PREFIX}_all_models.csv', index=False)
print(f"✓ Saved: {OUTPUT_PREFIX}_all_models.csv")

# ----------------------------------------------------------------------------
# Step 12: Summary Table
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 12: SUMMARY OF SIGNIFICANT RESULTS")
print("-" * 70)

summary_data = []
for model_name, results in all_results.items():
    for i, var_name in enumerate(results['X_cols']):
        if var_name != 'const' and 'Year' not in var_name:
            p_val = results['p_values'][i]
            summary_data.append({
                'Model': model_name,
                'Variable': var_name,
                'Coefficient': results['beta_corrected'][i],
                'Std_Error': results['beta_boot_std'][i],
                'P_Value': p_val,
                'Significant_5pct': p_val < 0.05,
                'Significant_1pct': p_val < 0.01
            })

if summary_data:
    summary_df = pd.DataFrame(summary_data)

    print("\n  Significant Variables (p < 0.05):")
    significant = summary_df[summary_df['Significant_5pct']]
    print(significant[['Model', 'Variable', 'Coefficient', 'P_Value']].to_string(index=False)
          if len(significant) else "  None")

    print("\n  Significant Variables (p < 0.01):")
    significant_1pct = summary_df[summary_df['Significant_1pct']]
    print(significant_1pct[['Model', 'Variable', 'Coefficient', 'P_Value']].to_string(index=False)
          if len(significant_1pct) else "  None")

    summary_df.to_csv(f'{OUTPUT_PREFIX}_summary.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_PREFIX}_summary.csv")
else:
    print("  No results to summarize")

# ----------------------------------------------------------------------------
# Step 13: Sample size consistency check across models
# ----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("STEP 13: SAMPLE SIZE PER MODEL (should all match - same df_clean)")
print("-" * 70)
for model_name, vars_list in models.items():
    available_vars = [v for v in vars_list if v in df_clean.columns]
    print(f"  {model_name}: N = {len(df_clean)}, variables = {available_vars}")

print("\n" + "=" * 70)
print("✓ STAGE 2 SIMAR-WILSON ANALYSIS COMPLETE!")
print("=" * 70)