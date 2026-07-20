"""
================================================================================
STAGE 2: SIMAR-WILSON DOUBLE-BOOTSTRAP TRUNCATED REGRESSION
================================================================================
UPDATED - Fixed numeric conversion issues
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

# Stage 1 DEA results
STAGE1_FILE = '/Users/aishaanibajaj/PycharmProjects/Internship/efficiency_scores_for_stage2.csv'

# WGI data file
WGI_FILE = '/Users/aishaanibajaj/Downloads/wgi_country_means.csv'

# Financial Development data file
FD_FILE = '/Users/aishaanibajaj/Downloads/FDI WHO.csv'

# Output file prefix
OUTPUT_PREFIX = 'simar_wilson_results'

# Bootstrap parameters
BOOTSTRAP_ITERATIONS = 500
SIGNIFICANCE_LEVEL = 0.05

# ============================================================================
# PART 1: TRUNCATED REGRESSION CLASS
# ============================================================================

class TruncatedRegression:
    """
    Truncated regression with lower bound 0 and upper bound 1
    """
    def __init__(self, lower_bound=0, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.params = None
        self.sigma = None

    def log_likelihood(self, params, X, y):
        """Negative log-likelihood for truncated normal"""
        beta = params[:-1]
        sigma = params[-1]

        if sigma <= 0:
            return 1e10

        mu = X @ beta
        z_lower = (self.lower_bound - mu) / sigma
        z_upper = (self.upper_bound - mu) / sigma

        ll = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma)**2
        ll -= np.log(stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower) + 1e-10)

        return -np.sum(ll)

    def fit(self, X, y, max_iter=1000):
        """Estimate truncated regression"""
        try:
            beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
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
    """
    Simar-Wilson double-bootstrap for truncated regression
    """
    n_obs, n_vars = X.shape

    # Step 1: Initial truncated regression
    model = TruncatedRegression(lower_bound=0, upper_bound=1)
    model.fit(X, y)
    beta_initial = model.params
    sigma_initial = model.sigma

    print(f"  Initial Sigma: {sigma_initial:.6f}")

    # Arrays to store bootstrap results
    beta_boot = np.zeros((B1, n_vars))

    print(f"\n  Running {B1} bootstrap iterations...")

    for i in range(B1):
        if (i + 1) % 100 == 0:
            print(f"    Iteration {i+1}/{B1}...")

        # Generate bootstrap sample
        mu = X @ beta_initial

        # Draw from truncated normal
        u_uniform = np.random.uniform(0, 1, n_obs)
        z_lower = (0 - mu) / sigma_initial
        z_upper = (1 - mu) / sigma_initial

        phi_lower = stats.norm.cdf(z_lower)
        phi_upper = stats.norm.cdf(z_upper)

        u_trunc = phi_lower + u_uniform * (phi_upper - phi_lower)
        u_trunc = np.clip(u_trunc, 1e-10, 1 - 1e-10)
        epsilon_boot = sigma_initial * stats.norm.ppf(u_trunc)

        # Generate bootstrapped efficiency scores
        y_boot = mu + epsilon_boot
        y_boot = np.clip(y_boot, 0.001, 0.999)

        # Re-estimate truncated regression
        boot_model = TruncatedRegression(lower_bound=0, upper_bound=1)
        try:
            boot_model.fit(X, y_boot, max_iter=500)
            beta_boot[i, :] = boot_model.params
        except:
            beta_boot[i, :] = beta_initial

    print("  Bootstrap complete!\n")

    # Calculate statistics
    beta_boot_mean = np.mean(beta_boot, axis=0)
    beta_boot_std = np.std(beta_boot, axis=0)

    # Bias correction
    bias = beta_boot_mean - beta_initial
    beta_corrected = beta_initial - bias

    # Percentile confidence intervals
    ci_lower = np.percentile(beta_boot, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(beta_boot, (1 - alpha/2) * 100, axis=0)

    # P-values
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
# PART 3: DATA LOADING AND PREPARATION FUNCTIONS
# ============================================================================

def load_wgi_data(file_path):
    """Load and prepare WGI data"""
    df = pd.read_csv(file_path)
    print(f"  WGI columns: {list(df.columns)}")
    return df

def load_fd_data(file_path):
    """Load and prepare FD data (wide format with years as columns)"""
    df = pd.read_csv(file_path)
    print(f"  FD columns: {list(df.columns)}")

    # Identify year columns (1980-2020)
    year_cols = [col for col in df.columns if col.isdigit() and int(col) >= 1980]

    if year_cols:
        print(f"  Found year columns: {year_cols[0]} to {year_cols[-1]}")

        # Melt the dataframe to long format
        id_vars = ['COUNTRY', 'INDICATOR', 'SERIES_CODE', 'SERIES_NAME']
        # Keep only columns that exist
        id_vars = [col for col in id_vars if col in df.columns]

        df_long = pd.melt(df,
                         id_vars=id_vars,
                         value_vars=year_cols,
                         var_name='Year',
                         value_name='FD_Value')

        # Convert Year to integer
        df_long['Year'] = df_long['Year'].astype(int)

        # Convert FD_Value to numeric, coercing errors to NaN
        df_long['FD_Value'] = pd.to_numeric(df_long['FD_Value'], errors='coerce')

        # Remove rows with NaN values
        df_long = df_long.dropna(subset=['FD_Value'])

        # Pivot to have indicators as columns
        if 'INDICATOR' in df_long.columns:
            # Use INDICATOR as the column names
            df_pivot = df_long.pivot_table(
                index=['COUNTRY', 'Year'],
                columns='INDICATOR',
                values='FD_Value'
            ).reset_index()

            # Rename columns to remove spaces and special characters
            df_pivot.columns = [col.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                               for col in df_pivot.columns]

            # Rename COUNTRY column to Country Name for consistency
            df_pivot.rename(columns={'COUNTRY': 'Country Name'}, inplace=True)

            print(f"  Created FD data with {len(df_pivot)} observations")
            print(f"  FD indicators: {[col for col in df_pivot.columns if col not in ['Country Name', 'Year']]}")
            return df_pivot
        else:
            print("  No INDICATOR column found, returning long format")
            return df_long
    else:
        print("  No year columns found (columns like 1980, 1981, etc.)")
        return df

# ============================================================================
# PART 4: MAIN ANALYSIS
# ============================================================================

print("="*70)
print("STAGE 2: SIMAR-WILSON DOUBLE-BOOTSTRAP TRUNCATED REGRESSION")
print("="*70)
print("\nFile paths being used:")
print(f"  Stage 1 DEA results: {STAGE1_FILE}")
print(f"  WGI data: {WGI_FILE}")
print(f"  FD data: {FD_FILE}")
print(f"  Output prefix: {OUTPUT_PREFIX}")
print(f"  Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")

# ----------------------------------------------------------------------------
# Step 1: Load Stage 1 Results
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 1: LOADING STAGE 1 RESULTS")
print("-"*70)

if not os.path.exists(STAGE1_FILE):
    print(f"\n❌ File not found: {STAGE1_FILE}")
    exit()

df_stage1 = pd.read_csv(STAGE1_FILE)
print(f"✓ Loaded: {STAGE1_FILE}")
print(f"  Observations: {len(df_stage1)}")
print(f"  Columns: {list(df_stage1.columns)}")

# Check for required columns
required_stage1 = ['DEA_Efficiency', 'Year']
missing = [col for col in required_stage1 if col not in df_stage1.columns]

if missing:
    print(f"\n❌ Missing columns in stage 1 file: {missing}")
    print("  Available columns:", list(df_stage1.columns))
    exit()

print(f"  Years: {df_stage1['Year'].min()} to {df_stage1['Year'].max()}")
print(f"  Countries: {df_stage1['Country Name'].nunique()} unique countries")

# ----------------------------------------------------------------------------
# Step 2: Load WGI Data
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 2: LOADING WGI DATA")
print("-"*70)

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

print("\n" + "-"*70)
print("STEP 3: LOADING FINANCIAL DEVELOPMENT DATA")
print("-"*70)

if not os.path.exists(FD_FILE):
    print(f"\n❌ File not found: {FD_FILE}")
    exit()

df_fd = load_fd_data(FD_FILE)
print(f"✓ Loaded and processed FD data")
print(f"  Observations: {len(df_fd)}")
print(f"  Columns: {list(df_fd.columns)}")

# ----------------------------------------------------------------------------
# Step 4: Merge All Data
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 4: MERGING ALL DATA")
print("-"*70)

# Ensure consistent column names
country_col_stage1 = 'Country Name'
country_col_wgi = 'Country Name'

# Check if FD has the right country column
if 'Country Name' not in df_fd.columns:
    print("  FD data columns:", list(df_fd.columns))
    print("  Renaming COUNTRY to Country Name for consistency...")
    df_fd.rename(columns={'COUNTRY': 'Country Name'}, inplace=True)

print(f"  Using country column: Country Name")

# Merge stage1 with WGI
df_merged = pd.merge(df_stage1, df_wgi, on=['Country Name'], how='inner')
print(f"  After WGI merge: {len(df_merged)} observations")

if len(df_merged) == 0:
    print("\n❌ No matches found with WGI data!")
    print(f"  Stage1 countries: {df_stage1['Country Name'].unique()[:10]}")
    print(f"  WGI countries: {df_wgi['Country Name'].unique()[:10]}")
    print("\n  Check country name consistency (case, spelling, abbreviations)")
    exit()

# Merge with FD (on country and year)
df_merged = pd.merge(df_merged, df_fd, on=['Country Name', 'Year'], how='inner')
print(f"  After FD merge: {len(df_merged)} observations")

if len(df_merged) == 0:
    print("\n❌ No matches found with FD data!")
    print(f"  Current countries: {df_merged['Country Name'].unique()[:10]}")
    print(f"  FD countries: {df_fd['Country Name'].unique()[:10]}")
    print(f"  Years in current data: {df_merged['Year'].unique()[:5]}")
    print(f"  Years in FD data: {df_fd['Year'].unique()[:5]}")
    print("\n  Check country name and year consistency")
    exit()

print(f"  Final merged data: {len(df_merged)} observations")
print(f"  Years: {df_merged['Year'].min()} to {df_merged['Year'].max()}")
print(f"  Countries: {df_merged['Country Name'].nunique()} unique countries")

# ----------------------------------------------------------------------------
# Step 5: Identify WGI and FD Columns
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 5: IDENTIFYING WGI AND FD COLUMNS")
print("-"*70)

all_cols = list(df_merged.columns)
print(f"  Available columns: {all_cols}")

# WGI columns (from your data)
wgi_columns = []
wgi_patterns = ['WGI_GE', 'WGI_CC', 'WGI_RL', 'WGI_composite']
for col in all_cols:
    if col in wgi_patterns or 'WGI' in col:
        wgi_columns.append(col)

# FD columns - exclude country, year, WGI columns, and stage1 columns
exclude_cols = ['Country Name', 'Year', 'DEA_Efficiency', 'FPE', 'CHE_PPP', 'GGHE_pct']
# Also exclude any columns that are not numeric
# Create FD Overall - only use numeric columns, excluding parent/mid-level indices
fd_parent_indices = ['Financial_Development_Index', 'Financial_Institutions_Index', 'Financial_Markets_Index']

if fd_columns:
    available_fd = [col for col in fd_columns if col in df_merged.columns]
    available_fd_leaves = [col for col in available_fd if col not in fd_parent_indices]

    if available_fd_leaves:
        for col in available_fd_leaves:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

        df_merged['FD_Overall'] = df_merged[available_fd_leaves].mean(axis=1)
        print(f"✓ FD_Overall rebuilt from {len(available_fd_leaves)} leaf-level components: {available_fd_leaves}")
    else:
        print("⚠️  No leaf-level FD columns found")
else:
    print("⚠️  No FD columns available")

print(f"  Identified WGI columns: {wgi_columns}")
print(f"  Identified FD columns: {fd_columns}")

# ----------------------------------------------------------------------------
# Step 6: Create Composite Indicators
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 6: CREATING COMPOSITE INDICATORS")
print("-"*70)

# WGI Composite is already in the data
if 'WGI_composite' in df_merged.columns:
    print(f"✓ WGI_Composite already exists")
else:
    # Create from components
    available_wgi = [col for col in wgi_columns if col in df_merged.columns and col != 'WGI_composite']
    if available_wgi:
        df_merged['WGI_Composite'] = df_merged[available_wgi].mean(axis=1)
        print(f"✓ WGI_Composite created from {len(available_wgi)} components")

# Create FD Overall - only use numeric columns
if fd_columns:
    available_fd = [col for col in fd_columns if col in df_merged.columns]
    if available_fd:
        # Convert to numeric and ensure all are float
        for col in available_fd:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

        # Create FD_Overall
        df_merged['FD_Overall'] = df_merged[available_fd].mean(axis=1)
        print(f"✓ FD_Overall created from {len(available_fd)} components")
        print(f"  FD indicators used: {available_fd}")
    else:
        print("⚠️  No FD columns found")
else:
    print("⚠️  No FD columns available")

# ----------------------------------------------------------------------------
# Step 7: Clean Data
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 7: CLEANING DATA")
print("-"*70)

# Keep necessary columns
keep_cols = ['DEA_Efficiency', 'Year', 'WGI_Composite', 'FD_Overall']
keep_cols.extend([col for col in wgi_columns if col in df_merged.columns and col != 'WGI_composite'])
keep_cols.extend([col for col in fd_columns if col in df_merged.columns])

# Remove duplicates
keep_cols = list(set(keep_cols))

# Check if required columns exist
for col in ['DEA_Efficiency', 'Year', 'WGI_Composite', 'FD_Overall']:
    if col not in df_merged.columns:
        print(f"⚠️  {col} not found in data!")
        if col == 'WGI_Composite':
            print("  Creating synthetic WGI_Composite for testing...")
            df_merged['WGI_Composite'] = np.random.uniform(-2, 2, len(df_merged))
        elif col == 'FD_Overall':
            print("  Creating sythetic FD_Overall for testing...")
            df_merged['FD_Overall'] = np.random.uniform(0, 1, len(df_merged))

df_clean = df_merged[keep_cols].copy()

# Convert all columns to numeric where possible
for col in df_clean.columns:
    if col not in ['Country Name', 'Year']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop rows with missing values
df_clean = df_clean.dropna()
print(f"  Observations after cleaning: {len(df_clean)}")

if len(df_clean) < 10:
    print(f"\n⚠️  Very few observations ({len(df_clean)}) after cleaning.")
    print("  Non-missing counts:")
    for col in df_clean.columns:
        print(f"    {col}: {df_clean[col].notna().sum()}")
    print("\n  You may need to check your data quality.")
    exit()

# Create year dummies
year_dummies = pd.get_dummies(df_clean['Year'], prefix='Year', drop_first=True)
print(f"  Year dummies created: {len(year_dummies.columns)} years")

# ----------------------------------------------------------------------------
# Step 8: Define Models
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 8: MODEL SPECIFICATIONS")
print("-"*70)

# Get available WGI and FD columns
available_wgi = [col for col in wgi_columns if col in df_clean.columns and col != 'WGI_composite']
available_fd = [col for col in fd_columns if col in df_clean.columns]

# Model 1: Aggregated
model1_vars = ['WGI_Composite', 'FD_Overall']

# Model 2: WGI Components + FD Overall
model2_vars = available_wgi + ['FD_Overall']

# Model 3: WGI Composite + FD Components
model3_vars = ['WGI_Composite'] + available_fd_leaves
# Model 4: Fully Disaggregated
model4_vars = available_wgi + available_fd_leaves

models = {
    'Model_1_Aggregated': model1_vars,
    'Model_2_WGI_Components': model2_vars,
    'Model_3_FD_Components': model3_vars,
    'Model_4_Fully_Disaggregated': model4_vars
}

# Only keep models with at least one variable
models = {k: v for k, v in models.items() if len(v) > 0}

print(f"  Running {len(models)} models:")
for name, vars_list in models.items():
    available = [v for v in vars_list if v in df_clean.columns]
    print(f"    - {name}: {len(available)} variables")
    if len(available) < len(vars_list):
        missing = [v for v in vars_list if v not in df_clean.columns]
        print(f"      Missing: {missing}")

# ----------------------------------------------------------------------------
# Step 9: Run Simar-Wilson for Each Model
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("STEP 9: RUNNING SIMAR-WILSON BOOTSTRAP")
print("-"*70)

all_results = {}

for model_name, vars_list in models.items():
    print(f"\n{'='*70}")
    print(f"Running: {model_name}")
    print(f"{'='*70}")

    # Check which variables are available
    available_vars = [v for v in vars_list if v in df_clean.columns]
    missing_vars = [v for v in vars_list if v not in df_clean.columns]

    if missing_vars:
        print(f"⚠️  Missing variables: {missing_vars}")
        print(f"  Using available: {available_vars}")

    if not available_vars:
        print("  No variables available, skipping...")
        continue

    # Prepare X and y
    year_cols = [col for col in year_dummies.columns]
    X_cols = ['const'] + available_vars + year_cols

    X = np.column_stack([np.ones(len(df_clean))] +
                        [df_clean[v].values for v in available_vars] +
                        [year_dummies[col].values for col in year_cols])
    y = df_clean['DEA_Efficiency'].values

    # Clean any inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.5, posinf=0.5, neginf=0.5)

    print(f"  Observations: {len(y)}")
    print(f"  Variables: {len(X_cols)}")
    print(f"  Variable names: {X_cols}")

    # Run bootstrap
    results = simar_wilson_bootstrap(X, y, B1=BOOTSTRAP_ITERATIONS, alpha=SIGNIFICANCE_LEVEL)

    # Store with variable names
    results['X_cols'] = X_cols
    results['model_name'] = model_name
    all_results[model_name] = results

# ----------------------------------------------------------------------------
# Step 10: Display Results
# ----------------------------------------------------------------------------

print("\n" + "="*70)
print("STEP 10: RESULTS")
print("="*70)

for model_name, results in all_results.items():
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"  Bootstrap Iterations: {BOOTSTRAP_ITERATIONS}")
    print(f"  Sigma: {results['sigma_initial']:.6f}")
    print(f"\n  {'Variable':<35} {'Coefficient':>12} {'Std Error':>12} {'Bias':>12} "
          f"{f'{SIGNIFICANCE_LEVEL/2*100:.1f}% CI':>14} "
          f"{f'{(1-SIGNIFICANCE_LEVEL/2)*100:.1f}% CI':>14} {'P-value':>10}")
    print(f"  {'-'*130}")

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

print("\n" + "-"*70)
print("STEP 11: SAVING RESULTS")
print("-"*70)

# Save results for each model
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

# Save all results in one file
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

print("\n" + "-"*70)
print("STEP 12: SUMMARY OF SIGNIFICANT RESULTS")
print("-"*70)

summary_data = []
for model_name, results in all_results.items():
    for i, var_name in enumerate(results['X_cols']):
        if var_name not in ['const'] and 'Year' not in var_name:
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
    significant = summary_df[summary_df['Significant_5pct'] == True]
    if len(significant) > 0:
        print(significant[['Model', 'Variable', 'Coefficient', 'P_Value']].to_string(index=False))
    else:
        print("  None")

    print("\n  Significant Variables (p < 0.01):")
    significant_1pct = summary_df[summary_df['Significant_1pct'] == True]
    if len(significant_1pct) > 0:
        print(significant_1pct[['Model', 'Variable', 'Coefficient', 'P_Value']].to_string(index=False))
    else:
        print("  None")

    # Save summary
    summary_df.to_csv(f'{OUTPUT_PREFIX}_summary.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_PREFIX}_summary.csv")
else:
    print("  No results to summarize")

print("\n" + "="*70)
print("✓ STAGE 2 SIMAR-WILSON ANALYSIS COMPLETE!")
print("="*70)
print("\n  Output Files:")
print(f"    - {OUTPUT_PREFIX}_Model_*.csv (per model results)")
print(f"    - {OUTPUT_PREFIX}_all_models.csv (all models combined)")
print(f"    - {OUTPUT_PREFIX}_summary.csv (summary of significant variables)")
print("\n  Check the output files for detailed results.")
print("="*70)
print("\nSample size used per model:")
for model_name, vars_list in models.items():
    available_vars = [v for v in vars_list if v in df_clean.columns]
    print(f"  {model_name}: N = {len(df_clean)}, variables = {available_vars}")