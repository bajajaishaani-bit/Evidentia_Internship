import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# EQUATION 2: REGRESSION ANALYSIS WITH HDI CONTROLS
# Fixed version - handles all data types properly
# ============================================================

# Load the dataset
df = pd.read_csv("eq2_dataset_with_hdi.csv")
print("=" * 80)
print("EQUATION 2: Health Expenditure Residuals Analysis")
print("=" * 80)
print(f"\nDataset loaded: {len(df)} countries")

# ── 1. Clean and prepare data ──────────────────────────────────────────────
print("\n1. Preparing data...")

# Convert all variables to numeric, coercing errors to NaN
numeric_cols = ['WGI_composite', 'GINI_mean', 'HDI_Value', 'Life_Expectancy',
                'Expected_Schooling', 'Mean_Schooling', 'GNI_per_capita',
                'mean_residual', 'mean_che', 'mean_oop', 'mean_gov']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle HDI category dummies - ensure they're numeric (0/1)
hdi_cat_cols = [col for col in df.columns if col.startswith('HDI_Cat_')]
for col in hdi_cat_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Drop rows with missing key variables
df_clean = df.dropna(subset=['WGI_composite', 'GINI_mean', 'HDI_Value']).copy()
print(f"Countries with complete data: {len(df_clean)}")

# ── 2. Descriptive Statistics ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 80)

print("\nSummary statistics for key variables:")
key_vars = ['mean_residual', 'WGI_composite', 'GINI_mean', 'HDI_Value']
print(df_clean[key_vars].describe().round(3))

print("\nCorrelation with residuals:")
for var in ['WGI_composite', 'GINI_mean', 'HDI_Value']:
    corr = df_clean['mean_residual'].corr(df_clean[var])
    print(f"  {var:20s}: {corr:.3f}")

# ── 3. Regression Models ──────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. REGRESSION RESULTS")
print("=" * 80)

# Define dependent variable
y = df_clean['mean_residual']

# ── Model 1: Base model (WGI + GINI) ──────────────────────────────────────────
print("\n--- Model 1: Base Model (WGI + GINI) ---")
X1 = df_clean[['WGI_composite', 'GINI_mean']].astype(float)
X1 = sm.add_constant(X1)
model1 = sm.OLS(y, X1).fit()
print(model1.summary())

# ── Model 2: Add HDI (continuous) ────────────────────────────────────────────
print("\n--- Model 2: Add HDI (continuous) ---")
X2 = df_clean[['WGI_composite', 'GINI_mean', 'HDI_Value']].astype(float)
X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2).fit()
print(model2.summary())

# ── Model 3: Add HDI components ──────────────────────────────────────────────
print("\n--- Model 3: Add HDI components ---")
# Prepare data with HDI components
hdi_components = []
if 'Life_Expectancy' in df_clean.columns:
    hdi_components.append('Life_Expectancy')
if 'Mean_Schooling' in df_clean.columns:
    hdi_components.append('Mean_Schooling')
if 'Expected_Schooling' in df_clean.columns:
    hdi_components.append('Expected_Schooling')
if 'GNI_per_capita' in df_clean.columns:
    df_clean['ln_GNI'] = np.log(df_clean['GNI_per_capita'] + 1)
    hdi_components.append('ln_GNI')

# Drop rows with missing HDI components
df_comp = df_clean.dropna(subset=['WGI_composite', 'GINI_mean'] + hdi_components).copy()
print(f"Countries with complete HDI components: {len(df_comp)}")

if len(df_comp) > 30:  # Only run if we have enough observations
    y_comp = df_comp['mean_residual']
    X3 = df_comp[['WGI_composite', 'GINI_mean'] + hdi_components].astype(float)
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(y_comp, X3).fit()
    print(model3.summary())
else:
    print("Not enough observations for Model 3")
    model3 = None

# ── Model 4: Add HDI categories ──────────────────────────────────────────────
print("\n--- Model 4: Add HDI categories ---")
hdi_cat_cols = [col for col in df_clean.columns if col.startswith('HDI_Cat_')]
if hdi_cat_cols:
    # Ensure all category dummies are numeric and create complete cases
    df_cat = df_clean[['WGI_composite', 'GINI_mean'] + hdi_cat_cols].astype(float)
    df_cat = df_cat.dropna()

    if len(df_cat) > 30:
        X4 = sm.add_constant(df_cat)
        y4 = df_clean.loc[df_cat.index, 'mean_residual']
        model4 = sm.OLS(y4, X4).fit()
        print(model4.summary())

        # Also show which categories exist
        print("\nHDI Category distribution:")
        print(df_clean['HDI_Category'].value_counts())
    else:
        print("Not enough observations for Model 4")
        model4 = None
else:
    print("No HDI category dummies found")
    model4 = None

# ── Model 5: Interaction terms ──────────────────────────────────────────────
print("\n--- Model 5: Add interaction (HDI * WGI) ---")
df_clean['HDI_WGI'] = df_clean['HDI_Value'] * df_clean['WGI_composite']
X5 = df_clean[['WGI_composite', 'GINI_mean', 'HDI_Value', 'HDI_WGI']].astype(float)
X5 = sm.add_constant(X5)
model5 = sm.OLS(y, X5).fit()
print(model5.summary())

# ── Model 6: Robust standard errors (HC1) ────────────────────────────────────
print("\n--- Model 6: Model 2 with Robust Standard Errors ---")
model6 = model2.get_robustcov_results(cov_type='HC1')
print(model6.summary())

# ── 4. Model Comparison ──────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. MODEL COMPARISON")
print("=" * 80)

models = [model1, model2]
model_names = ['Base (WGI+GINI)', 'Add HDI']

if model3 is not None:
    models.append(model3)
    model_names.append('Add HDI Components')
if model4 is not None:
    models.append(model4)
    model_names.append('Add HDI Categories')
models.append(model5)
model_names.append('Interaction (HDI*WGI)')

print("\nModel\t\t\t\tR²\t\tAdj R²\t\tAIC\t\tBIC")
print("-" * 90)
for name, model in zip(model_names, models):
    if model is not None:
        print(f"{name:25s}\t{model.rsquared:.4f}\t{model.rsquared_adj:.4f}\t{model.aic:.1f}\t{model.bic:.1f}")

# F-test for model improvement
print("\n" + "-" * 90)
print("F-test: Is adding HDI statistically significant?")
print("(Comparing Model 2 vs Model 1)")
if model1 and model2:
    f_stat = ((model1.ssr - model2.ssr) / (model2.df_model - model1.df_model)) / (model2.ssr / model2.df_resid)
    p_value = 1 - stats.f.cdf(f_stat, model2.df_model - model1.df_model, model2.df_resid)
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✅ HDI is statistically significant (p < 0.05)")
    else:
        print(f"  ❌ HDI is NOT statistically significant (p = {p_value:.3f})")

# ── 5. Diagnostic Tests ──────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5. DIAGNOSTIC TESTS")
print("=" * 80)

if model2 is not None:
    # Test for heteroskedasticity
    print("\nBreusch-Pagan test for heteroskedasticity:")
    try:
        bp_test = sm.stats.diagnostic.het_breuschpagan(model2.resid, model2.model.exog)
        print(f"  LM-statistic: {bp_test[0]:.4f}")
        print(f"  LM p-value: {bp_test[1]:.4f}")
        if bp_test[1] < 0.05:
            print("  ⚠️  Heteroskedasticity present - use robust standard errors (Model 6)")
        else:
            print("  ✅ No heteroskedasticity detected")
    except:
        print("  Could not compute Breusch-Pagan test")

    # Test for normality
    print("\nJarque-Bera test for normality of residuals:")
    try:
        jb_test = sm.stats.diagnostic.jarque_bera(model2.resid)
        print(f"  JB-statistic: {jb_test[0]:.4f}")
        print(f"  JB p-value: {jb_test[1]:.4f}")
        if jb_test[1] < 0.05:
            print("  ⚠️  Residuals not normal (consider robust inference)")
        else:
            print("  ✅ Residuals appear normal")
    except:
        print("  Could not compute Jarque-Bera test")

# ── 6. Visualization ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("6. VISUALIZATION")
print("=" * 80)

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Scatter: HDI vs Residuals
ax1 = axes[0, 0]
ax1.scatter(df_clean['HDI_Value'], df_clean['mean_residual'], alpha=0.6)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('HDI Value')
ax1.set_ylabel('Residual')
ax1.set_title('HDI vs Health Expenditure Residual')
ax1.grid(True, alpha=0.3)

# Scatter: WGI vs Residuals
ax2 = axes[0, 1]
ax2.scatter(df_clean['WGI_composite'], df_clean['mean_residual'], alpha=0.6)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('WGI Composite')
ax2.set_ylabel('Residual')
ax2.set_title('Governance vs Residual')
ax2.grid(True, alpha=0.3)

# Scatter: GINI vs Residuals
ax3 = axes[0, 2]
ax3.scatter(df_clean['GINI_mean'], df_clean['mean_residual'], alpha=0.6)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3.set_xlabel('GINI Index')
ax3.set_ylabel('Residual')
ax3.set_title('Inequality vs Residual')
ax3.grid(True, alpha=0.3)

# Residuals distribution
ax4 = axes[1, 0]
ax4.hist(model2.resid, bins=20, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Residuals')
ax4.grid(True, alpha=0.3)

# Q-Q plot
ax5 = axes[1, 1]
stats.probplot(model2.resid, dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot')

# Residuals vs Fitted
ax6 = axes[1, 2]
ax6.scatter(model2.fittedvalues, model2.resid, alpha=0.6)
ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax6.set_xlabel('Fitted Values')
ax6.set_ylabel('Residuals')
ax6.set_title('Residuals vs Fitted')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eq2_hdi_regression_results.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'eq2_hdi_regression_results.png'")
plt.show()

# ── 7. Countries with largest residuals ──────────────────────────────────────
print("\n" + "=" * 80)
print("7. COUNTRIES WITH LARGEST RESIDUALS")
print("=" * 80)

print("\nTop 10 Underperformers (Actual CHE < Predicted CHE):")
underperformers = df_clean.nsmallest(10, 'mean_residual')
print(underperformers[['Country Name', 'mean_residual', 'HDI_Value',
                       'WGI_composite', 'GINI_mean']].to_string(index=False))

print("\nTop 10 Overperformers (Actual CHE > Predicted CHE):")
overperformers = df_clean.nlargest(10, 'mean_residual')
print(overperformers[['Country Name', 'mean_residual', 'HDI_Value',
                      'WGI_composite', 'GINI_mean']].to_string(index=False))

# ── 8. Save Results ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("8. SAVING RESULTS")
print("=" * 80)

# Create results summary
results_data = []
for name, model in zip(model_names, models):
    if model is not None:
        results_data.append({
            'Model': name,
            'R_Squared': model.rsquared,
            'Adj_R_Squared': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'N_Obs': model.nobs,
            'WGI_Coef': model.params.get('WGI_composite', np.nan),
            'WGI_Pval': model.pvalues.get('WGI_composite', np.nan),
            'GINI_Coef': model.params.get('GINI_mean', np.nan),
            'GINI_Pval': model.pvalues.get('GINI_mean', np.nan),
            'HDI_Coef': model.params.get('HDI_Value', np.nan),
            'HDI_Pval': model.pvalues.get('HDI_Value', np.nan)
        })

results_df = pd.DataFrame(results_data)
results_df.to_csv('eq2_hdi_regression_summary.csv', index=False)
print("Results saved as 'eq2_hdi_regression_summary.csv'")

# Save the cleaned dataset
df_clean.to_csv('eq2_dataset_clean.csv', index=False)
print("Cleaned dataset saved as 'eq2_dataset_clean.csv'")

print("\n" + "=" * 80)
print("REGRESSION ANALYSIS COMPLETE!")
print("=" * 80)

# Print key findings
if model2 is not None:
    print(f"\n📊 KEY FINDINGS:")
    print(f"  • Base model R²: {model1.rsquared:.4f}")
    print(f"  • Model with HDI R²: {model2.rsquared:.4f}")
    print(f"  • Additional R² from HDI: {model2.rsquared - model1.rsquared:.4f}")
    print(f"  • HDI coefficient: {model2.params.get('HDI_Value', 0):.4f}")
    print(f"  • HDI p-value: {model2.pvalues.get('HDI_Value', 0):.4f}")

    if model2.pvalues.get('HDI_Value', 1) < 0.05:
        print(f"  ✅ HDI is a statistically significant predictor")
    else:
        print(f"  ❌ HDI is NOT statistically significant (p > 0.05)")