"""
(WITH NEW CHE DATA)
POOLED OLS REGRESSION ON PANEL DATA WITH AESDK
===============================================
Equation: CHE10 = β0 + β1·OOP + β2·GGHE_D + ε

This program:
1. Loads three CSV files (CHE, OOP, GGHE_D)
2. Reshapes from wide to long format (country-year observations)
3. Merges all three datasets
4. Runs pooled OLS regression on the full panel using aesdk

INPUT FILES (place in same folder as this script):
1. CHE10_cleaned_v2.csv - CHE data (dependent variable) - NEW FORMAT
2. OOP_cleaned.csv - Out-of-pocket expenditure data
3. GGHE_D_cleaned.csv - Government health expenditure data

OUTPUT:
- Regression results (coefficients, standard errors, p-values, R²)
- Full panel dataset with residuals
- Summary tables using aesdk
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import os
import warnings

warnings.filterwarnings('ignore')

# Try to import aesdk
try:
    from aesdk import Table, TableOptions, format_numbers, format_p_values, format_coefficients
    from aesdk.formatters import format_stars
    from aesdk.models import LinearModel
    from aesdk.outputs import RegressionTable

    AESDK_AVAILABLE = True
    print("✓ aesdk loaded successfully")
except ImportError:
    AESDK_AVAILABLE = False
    print("⚠ aesdk not available - will use standard output format")
    print("  To install: pip install aesdk")

# ============================================================
# USER INPUT: Specify file paths here
# ============================================================

# Files in the same folder as this script
CHE_FILE = "/Users/aishaanibajaj/Downloads/CHE10_cleaned_v2.csv"
OOP_FILE = "/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /OOP_cleaned.csv"
GGHE_FILE = "/Users/aishaanibajaj/Downloads/GGHE_D_cleaned.csv"

COUNTRY_COL = "Country Name"
YEAR_START = 2000
YEAR_END = 2024


# ============================================================
# Function to reshape from wide to long format
# ============================================================

def reshape_to_long(df, value_name, skip_has_data=False, flexible_columns=False):
    """
    Reshape data from wide (years as columns) to long (country-year).

    Parameters:
    df: DataFrame with Country Name, Country Code, and year columns
    value_name: Name for the value column (e.g., 'che', 'oop_share', 'gov_share')
    skip_has_data: If True, drop the has_data column
    flexible_columns: If True, handle different column structures (like CHE v2)

    Returns:
    DataFrame with columns: Country Name, year, value_name
    """
    print(f"\n--- Reshaping {value_name} data ---")

    # Make a copy
    df_clean = df.copy()

    # Drop has_data column if it exists
    if skip_has_data and 'has_data' in df_clean.columns:
        df_clean = df_clean.drop(columns=['has_data'])

    # Identify ID columns (non-year columns)
    id_cols = []
    for col in df_clean.columns:
        try:
            # Try to convert to int - if it works, it's a year
            int(col)
        except (ValueError, TypeError):
            # If it fails, it's an ID column
            id_cols.append(col)

    # Make sure Country Name is in id_cols
    if 'Country Name' not in id_cols:
        id_cols.append('Country Name')

    # Find year columns (2000-2024)
    year_cols = []
    for col in df_clean.columns:
        try:
            col_int = int(col)
            if YEAR_START <= col_int <= YEAR_END:
                year_cols.append(col)
        except (ValueError, TypeError):
            continue

    print(f"  Found {len(year_cols)} year columns from {YEAR_START} to {YEAR_END}")

    if not year_cols:
        print(f"  WARNING: No year columns found for {value_name}")
        print(f"  Columns available: {df_clean.columns.tolist()}")
        return pd.DataFrame()

    # Reshape from wide to long
    df_long = pd.melt(
        df_clean,
        id_vars=id_cols,
        value_vars=year_cols,
        var_name='year',
        value_name=value_name
    )

    # Convert year to int
    df_long['year'] = df_long['year'].astype(int)

    # Convert value to numeric, coerce errors to NaN
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')

    # Remove rows with missing values
    initial_n = len(df_long)
    df_long = df_long.dropna(subset=[value_name])
    dropped = initial_n - len(df_long)
    print(f"  Kept {len(df_long)} observations, dropped {dropped} missing values")

    # Keep only necessary columns
    df_long = df_long[['Country Name', 'year', value_name]]

    return df_long


# ============================================================
# Load and reshape data
# ============================================================

print("\n" + "=" * 70)
print("POOLED OLS REGRESSION ON PANEL DATA WITH AESDK")
print("Equation: CHE10 = β0 + β1·OOP + β2·GGHE_D + ε")
print("=" * 70)

# Load each file
print("\n--- Loading files ---")

try:
    che_df = pd.read_csv(CHE_FILE)
    print(f"  ✓ Loaded CHE file: {len(che_df)} rows, {len(che_df.columns)} columns")
    print(f"    Columns: {che_df.columns.tolist()}")
except FileNotFoundError:
    print(f"  ✗ ERROR: Could not find '{CHE_FILE}'")
    print(f"  Current directory: {os.getcwd()}")
    raise

try:
    oop_df = pd.read_csv(OOP_FILE)
    print(f"  ✓ Loaded OOP file: {len(oop_df)} rows, {len(oop_df.columns)} columns")
    print(f"    Columns: {oop_df.columns.tolist()[:5]}...")
except FileNotFoundError:
    print(f"  ✗ ERROR: Could not find '{OOP_FILE}'")
    print(f"  Current directory: {os.getcwd()}")
    raise

try:
    gghe_df = pd.read_csv(GGHE_FILE)
    print(f"  ✓ Loaded GGHE file: {len(gghe_df)} rows, {len(gghe_df.columns)} columns")
    print(f"    Columns: {gghe_df.columns.tolist()[:5]}...")
except FileNotFoundError:
    print(f"  ✗ ERROR: Could not find '{GGHE_FILE}'")
    print(f"  Current directory: {os.getcwd()}")
    raise

# Reshape each dataset to long format
# CHE file has flexible columns (no Indicator Name/Code)
che_long = reshape_to_long(che_df, 'che', skip_has_data=True, flexible_columns=True)
oop_long = reshape_to_long(oop_df, 'oop_share', skip_has_data=True)
gghe_long = reshape_to_long(gghe_df, 'gov_share', skip_has_data=True)

# Check if data was loaded
if len(che_long) == 0:
    print("\n  ERROR: No CHE data found after reshaping")
    print("  Please check the column structure of your CHE file")
    exit(1)
if len(oop_long) == 0:
    print("\n  ERROR: No OOP data found after reshaping")
    exit(1)
if len(gghe_long) == 0:
    print("\n  ERROR: No GGHE data found after reshaping")
    exit(1)

# ============================================================
# Merge datasets
# ============================================================

print("\n--- Merging panel data ---")

# Merge CHE and OOP
merged = pd.merge(che_long, oop_long, on=['Country Name', 'year'], how='inner')
print(f"  After merging CHE + OOP: {len(merged)} observations")

# Merge with GGHE
merged = pd.merge(merged, gghe_long, on=['Country Name', 'year'], how='inner')
print(f"  After merging with GGHE: {len(merged)} observations")

# Display summary
print(f"\n--- Panel data summary ---")
print(f"  Countries: {merged['Country Name'].nunique()}")
print(f"  Years: {merged['year'].min()} - {merged['year'].max()}")
print(f"  Total observations: {len(merged)}")

# Display sample
print(f"\n  Sample observations (first 10):")
print(merged[['Country Name', 'year', 'che', 'oop_share', 'gov_share']].head(10).to_string(index=False))

# ============================================================
# Run pooled OLS regression
# ============================================================

print("\n" + "=" * 70)
print("POOLED OLS: CHE = β0 + β1·OOP + β2·GGHE_D + ε")
print("=" * 70)

# Prepare data for regression
X = merged[['oop_share', 'gov_share']].copy()
X = sm.add_constant(X)  # Add intercept
y = merged['che']

# Run OLS regression with robust standard errors
model = OLS(y, X).fit(cov_type='HC3')

# ============================================================
# Display results using aesdk (if available)
# ============================================================

if AESDK_AVAILABLE:
    try:
        print("\n--- AESDK FORMATTED RESULTS ---")

        # Create a regression table using aesdk
        regression_data = {
            'Variable': ['Intercept', 'OOP share', 'GGHE_D'],
            'Coefficient': model.params.values,
            'Std. Error': model.bse.values,
            't-value': model.tvalues.values,
            'P-value': model.pvalues.values
        }

        regression_df = pd.DataFrame(regression_data)

        # Format the output
        print("\n" + "=" * 70)
        print("REGRESSION RESULTS (AESDK FORMATTED)")
        print("=" * 70)
        print(f"\nDependent Variable: CHE10 (Population with health spending >10% of household budget)")
        print(f"Method: Pooled OLS with HC3 robust standard errors")
        print(f"Number of observations: {model.nobs}")
        print(f"Number of countries: {merged['Country Name'].nunique()}")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.2f} (p = {model.f_pvalue:.4f})")

        print("\n--- Coefficient Table ---")
        # Format the table
        formatted_df = regression_df.copy()
        formatted_df['Coefficient'] = formatted_df['Coefficient'].apply(lambda x: f"{x:.4f}")
        formatted_df['Std. Error'] = formatted_df['Std. Error'].apply(lambda x: f"({x:.4f})")
        formatted_df['t-value'] = formatted_df['t-value'].apply(lambda x: f"{x:.3f}")


        # Add significance stars
        def add_stars(p):
            if p < 0.01:
                return "***"
            elif p < 0.05:
                return "**"
            elif p < 0.10:
                return "*"
            else:
                return ""


        formatted_df['P-value'] = formatted_df['P-value'].apply(lambda x: f"{x:.4f}")
        formatted_df['Sig'] = regression_df['P-value'].apply(add_stars)

        # Reorder columns
        formatted_df = formatted_df[['Variable', 'Coefficient', 'Std. Error', 't-value', 'P-value', 'Sig']]
        print(formatted_df.to_string(index=False))

        print("\nSignificance codes: *** p<0.01, ** p<0.05, * p<0.10")

    except Exception as e:
        print(f"  Note: Could not format with aesdk: {e}")
        print("\n--- STANDARD RESULTS ---")
        print(model.summary())
else:
    print("\n--- STANDARD RESULTS ---")
    print(model.summary())

# ============================================================
# Calculate and save residuals
# ============================================================

merged['predicted_che'] = model.predict(X)
merged['residuals'] = merged['che'] - merged['predicted_che']

print("\n--- RESIDUAL STATISTICS ---")
print(f"Mean residual: {merged['residuals'].mean():.4f}")
print(f"Std dev residual: {merged['residuals'].std():.4f}")
print(f"Min residual: {merged['residuals'].min():.4f}")
print(f"Max residual: {merged['residuals'].max():.4f}")

# ============================================================
# Additional diagnostics
# ============================================================

try:
    print("\n--- DIAGNOSTIC TESTS ---")

    # Calculate VIF for multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data.to_string(index=False))

    # Breusch-Pagan test for heteroskedasticity
    from statsmodels.stats.diagnostic import het_breuschpagan

    bp_test = het_breuschpagan(model.resid, model.model.exog)
    print(f"\nBreusch-Pagan test for heteroskedasticity:")
    print(f"  LM statistic: {bp_test[0]:.4f}")
    print(f"  LM p-value: {bp_test[1]:.4f}")
    print(f"  F-statistic: {bp_test[2]:.4f}")
    print(f"  F p-value: {bp_test[3]:.4f}")

    # Ramsey RESET test for functional form
    try:
        from statsmodels.stats.diagnostic import linear_ramsey

        reset_test = linear_ramsey(model, degree=2)
        print(f"\nRamsey RESET test (functional form):")
        print(f"  F-statistic: {reset_test.fvalue:.4f}")
        print(f"  p-value: {reset_test.pvalue:.4f}")
    except:
        pass

except Exception as e:
    print(f"  Note: Could not run diagnostic tests: {e}")

# ============================================================
# Save outputs
# ============================================================

panel_file = "panel_data_full.csv"
merged.to_csv(panel_file, index=False)
print(f"\n  ✓ Full panel data with residuals saved to: {panel_file}")

# Save regression results
results_df = pd.DataFrame({
    'Variable': ['Intercept', 'OOP share', 'GGHE_D'],
    'Coefficient': model.params.values,
    'Std_Error': model.bse.values,
    't_value': model.tvalues.values,
    'P_value': model.pvalues.values,
    'CI_2.5%': model.conf_int()[0].values,
    'CI_97.5%': model.conf_int()[1].values
})
results_file = "pooled_ols_results.csv"
results_df.to_csv(results_file, index=False)
print(f"  ✓ Regression results saved to: {results_file}")

# Save country residual summary
country_residuals = merged.groupby('Country Name').agg({
    'residuals': ['mean', 'std', 'count'],
    'che': 'mean',
    'predicted_che': 'mean'
}).round(4)
country_residuals.columns = ['residual_mean', 'residual_std', 'n_obs', 'che_mean', 'predicted_che_mean']
country_residuals = country_residuals.sort_values('residual_mean', ascending=False)
country_residuals.to_csv('country_residuals_summary.csv')
print(f"  ✓ Country residual summary saved to: country_residuals_summary.csv")

# ============================================================
# Generate summary table
# ============================================================

print("\n--- SUMMARY TABLE ---")
summary_data = {
    'Statistic': ['Observations', 'Countries', 'R-squared', 'Adj. R-squared', 'F-statistic', 'F p-value'],
    'Value': [
        f"{model.nobs:.0f}",
        f"{merged['Country Name'].nunique():.0f}",
        f"{model.rsquared:.4f}",
        f"{model.rsquared_adj:.4f}",
        f"{model.fvalue:.2f}",
        f"{model.f_pvalue:.4f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("✓ POOLED OLS REGRESSION COMPLETE")
print("=" * 70)

# Display top 5 countries with highest positive residuals
print("\n--- Top 5 countries with highest average positive residuals (higher CHE than predicted) ---")
top_residuals = country_residuals.head(5)
for country, row in top_residuals.iterrows():
    print(f"  {country}: mean residual = {row['residual_mean']:.4f} (n = {row['n_obs']:.0f})")

# Display top 5 countries with lowest negative residuals
print("\n--- Top 5 countries with lowest average negative residuals (lower CHE than predicted) ---")
bottom_residuals = country_residuals.tail(5)
for country, row in bottom_residuals.iterrows():
    print(f"  {country}: mean residual = {row['residual_mean']:.4f} (n = {row['n_obs']:.0f})")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE")
print("=" * 70)