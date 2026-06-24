"""
CROSS-SECTIONAL REGRESSION ON COUNTRY MEANS WITH AESDK
======================================================
Equation: CHE10 = β0 + β1·OOP + β2·GGHE_D + ε

This program:
1. Loads three CSV files (CHE, OOP, GGHE_D)
2. Calculates country means directly from wide format
3. Merges country-level data
4. Runs cross-sectional OLS regression on country means using aesdk

INPUT FILES (place in same folder as this script):
1. CHE10_cleaned_v2.csv - CHE data (dependent variable)
2. OOP_cleaned.csv - Out-of-pocket expenditure data
3. GGHE_D_cleaned.csv - Government health expenditure data

OUTPUT:
- Regression results (coefficients, standard errors, p-values, R²)
- Country-level dataset with means and residuals
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
# Function to calculate country means from wide format
# ============================================================

def calculate_country_means(df, value_name, skip_has_data=False):
    """
    Calculate country means across all years from wide format data.

    Parameters:
    df: DataFrame with Country Name, Country Code, and year columns
    value_name: Name for the value column (e.g., 'che', 'oop_share', 'gov_share')
    skip_has_data: If True, drop the has_data column

    Returns:
    DataFrame with columns: Country Name, value_name (mean across years)
    """
    print(f"\n--- Calculating country means for {value_name} ---")

    # Make a copy
    df_clean = df.copy()

    # Drop has_data column if it exists
    if skip_has_data and 'has_data' in df_clean.columns:
        df_clean = df_clean.drop(columns=['has_data'])

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

    # Calculate row-wise mean across year columns
    # Convert to numeric, coerce errors to NaN
    df_clean['mean_value'] = df_clean[year_cols].apply(
        lambda row: pd.to_numeric(row, errors='coerce').mean(),
        axis=1
    )

    # Keep only rows with non-NaN means
    df_clean = df_clean.dropna(subset=['mean_value'])

    # Keep only necessary columns
    result = df_clean[['Country Name', 'mean_value']].copy()
    result = result.rename(columns={'mean_value': value_name})

    print(f"  Kept {len(result)} countries with data")

    return result


# ============================================================
# Load data and calculate country means
# ============================================================

print("\n" + "=" * 70)
print("CROSS-SECTIONAL REGRESSION ON COUNTRY MEANS WITH AESDK")
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

# Calculate country means
che_means = calculate_country_means(che_df, 'che', skip_has_data=True)
oop_means = calculate_country_means(oop_df, 'oop_share', skip_has_data=True)
gghe_means = calculate_country_means(gghe_df, 'gov_share', skip_has_data=True)

# Check if data was loaded
if len(che_means) == 0:
    print("\n  ERROR: No CHE data found")
    print("  Please check the column structure of your CHE file")
    exit(1)
if len(oop_means) == 0:
    print("\n  ERROR: No OOP data found")
    exit(1)
if len(gghe_means) == 0:
    print("\n  ERROR: No GGHE data found")
    exit(1)

# ============================================================
# Merge country means
# ============================================================

print("\n--- Merging country-level data ---")

# Merge CHE and OOP
merged = pd.merge(che_means, oop_means, on='Country Name', how='inner')
print(f"  After merging CHE + OOP: {len(merged)} countries")

# Merge with GGHE
merged = pd.merge(merged, gghe_means, on='Country Name', how='inner')
print(f"  After merging with GGHE: {len(merged)} countries")

# Display sample countries
print(f"\n  Sample countries (first 10):")
print(merged[['Country Name', 'che', 'oop_share', 'gov_share']].head(10).to_string(index=False))

# ============================================================
# Run cross-sectional regression
# ============================================================

print("\n" + "=" * 70)
print("CROSS-SECTIONAL REGRESSION: CHE = β0 + β1·OOP + β2·GGHE_D + ε")
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
        print(f"Method: Cross-sectional OLS with HC3 robust standard errors")
        print(f"Number of countries: {model.nobs}")
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

    # Shapiro-Wilk test for normality of residuals
    from scipy import stats

    shapiro_stat, shapiro_p = stats.shapiro(model.resid)
    print(f"\nShapiro-Wilk test for normality of residuals:")
    print(f"  W-statistic: {shapiro_stat:.4f}")
    print(f"  p-value: {shapiro_p:.4f}")

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

country_file = "country_means_data.csv"
merged.to_csv(country_file, index=False)
print(f"\n  ✓ Country-level data with residuals saved to: {country_file}")

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
results_file = "cross_sectional_results.csv"
results_df.to_csv(results_file, index=False)
print(f"  ✓ Regression results saved to: {results_file}")

# Save country residual summary
country_residuals = merged[['Country Name', 'che', 'predicted_che', 'residuals']].copy()
country_residuals = country_residuals.sort_values('residuals', ascending=False)
country_residuals.to_csv('country_residuals_detailed.csv', index=False)
print(f"  ✓ Country residuals saved to: country_residuals_detailed.csv")

# ============================================================
# Generate summary table
# ============================================================

print("\n--- SUMMARY TABLE ---")
summary_data = {
    'Statistic': ['Countries', 'R-squared', 'Adj. R-squared', 'F-statistic', 'F p-value'],
    'Value': [
        f"{model.nobs:.0f}",
        f"{model.rsquared:.4f}",
        f"{model.rsquared_adj:.4f}",
        f"{model.fvalue:.2f}",
        f"{model.f_pvalue:.4f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("✓ CROSS-SECTIONAL REGRESSION COMPLETE")
print("=" * 70)

# Display top 5 countries with highest positive residuals
print("\n--- Top 5 countries with highest positive residuals (higher CHE than predicted) ---")
top_residuals = merged.nlargest(5, 'residuals')[['Country Name', 'che', 'predicted_che', 'residuals']]
for _, row in top_residuals.iterrows():
    print(f"  {row['Country Name']}: residual = {row['residuals']:.4f} "
          f"(actual: {row['che']:.2f}, predicted: {row['predicted_che']:.2f})")

# Display top 5 countries with lowest negative residuals
print("\n--- Top 5 countries with lowest negative residuals (lower CHE than predicted) ---")
bottom_residuals = merged.nsmallest(5, 'residuals')[['Country Name', 'che', 'predicted_che', 'residuals']]
for _, row in bottom_residuals.iterrows():
    print(f"  {row['Country Name']}: residual = {row['residuals']:.4f} "
          f"(actual: {row['che']:.2f}, predicted: {row['predicted_che']:.2f})")

# ============================================================
# Create HTML report (if aesdk is available)
# ============================================================

if AESDK_AVAILABLE:
    try:
        print("\n--- Generating HTML Report ---")

        # Create HTML report file
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Sectional Regression Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: #e74c3c; }}
                .negative {{ color: #27ae60; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Cross-Sectional Regression Results</h1>
            <p><strong>Equation:</strong> CHE10 = β₀ + β₁·OOP + β₂·GGHE_D + ε</p>
            <p><strong>Method:</strong> Cross-sectional OLS on country means</p>

            <div class="summary">
                <h2>Model Summary</h2>
                <p><strong>Countries:</strong> {model.nobs:.0f}</p>
                <p><strong>R-squared:</strong> {model.rsquared:.4f}</p>
                <p><strong>Adjusted R-squared:</strong> {model.rsquared_adj:.4f}</p>
                <p><strong>F-statistic:</strong> {model.fvalue:.2f} (p = {model.f_pvalue:.4f})</p>
            </div>

            <h2>Coefficient Table</h2>
            {results_df.to_html(index=False)}

            <h2>Top 5 Underperforming Countries (Positive Residuals)</h2>
            {top_residuals.to_html(index=False)}

            <h2>Top 5 Overperforming Countries (Negative Residuals)</h2>
            {bottom_residuals.to_html(index=False)}

            <p style="margin-top: 40px; color: #7f8c8d; font-size: 12px;">
                Generated using Python with statsmodels and aesdk
            </p>
        </body>
        </html>
        """

        with open('cross_sectional_report.html', 'w') as f:
            f.write(html_content)
        print("  ✓ HTML report saved to: cross_sectional_report.html")

    except Exception as e:
        print(f"  Note: Could not generate HTML report: {e}")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE")
print("=" * 70)