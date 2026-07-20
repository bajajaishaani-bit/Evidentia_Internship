"""
================================================================================
FINANCIAL PROTECTION EFFICIENCY (FPE) - DEA ANALYSIS
With Correct Model Specification
================================================================================
INPUTS:  CHE_PPP, OOP_pct
OUTPUTS: FPE (100 - CHE10)
Model:   VRS Input-Oriented
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DEA CLASS
# ============================================================================

class DEA:
    """
    Data Envelopment Analysis for Financial Protection Efficiency
    """

    def __init__(self, inputs, outputs, returns='VRS', orientation='input'):
        self.inputs = np.array(inputs, dtype=float)
        self.outputs = np.array(outputs, dtype=float)
        self.n_dmus = len(inputs)
        self.n_inputs = self.inputs.shape[1] if len(self.inputs.shape) > 1 else 1
        self.n_outputs = self.outputs.shape[1] if len(self.outputs.shape) > 1 else 1
        self.returns = returns
        self.orientation = orientation
        self.efficiency_scores = None
        self.lambdas = None
        self.slack_input = None
        self.slack_output = None

    def _validate_data(self):
        if np.any(self.inputs <= 0):
            raise ValueError("All inputs must be positive")
        if np.any(self.outputs <= 0):
            raise ValueError("All outputs must be positive")
        if self.n_dmus != len(self.outputs):
            raise ValueError("Number of DMUs must match for inputs and outputs")

    def _solve_input_oriented(self, k):
        n_vars = 1 + self.n_dmus
        c = np.zeros(n_vars)
        c[0] = 1.0

        A_ub = []
        b_ub = []

        for i in range(self.n_inputs):
            row = np.zeros(n_vars)
            row[0] = -self.inputs[k, i]
            for j in range(self.n_dmus):
                row[1 + j] = self.inputs[j, i]
            A_ub.append(row)
            b_ub.append(0.0)

        for r in range(self.n_outputs):
            row = np.zeros(n_vars)
            row[0] = 0.0
            for j in range(self.n_dmus):
                row[1 + j] = -self.outputs[j, r]
            A_ub.append(row)
            b_ub.append(-self.outputs[k, r])

        if self.returns == 'VRS':
            A_eq = np.zeros((1, n_vars))
            A_eq[0, 0] = 0.0
            for j in range(self.n_dmus):
                A_eq[0, 1 + j] = 1.0
            b_eq = np.array([1.0])
        else:
            A_eq = None
            b_eq = None

        bounds = [(0, None)] * n_vars

        result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                        A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result.success:
            return 1.0, np.zeros(self.n_dmus), np.zeros(self.n_inputs), np.zeros(self.n_outputs)

        theta = result.x[0]
        lambda_vec = result.x[1:]

        input_slack = np.zeros(self.n_inputs)
        output_slack = np.zeros(self.n_outputs)

        for i in range(self.n_inputs):
            lhs = sum(lambda_vec[j] * self.inputs[j, i] for j in range(self.n_dmus))
            input_slack[i] = self.inputs[k, i] * theta - lhs

        for r in range(self.n_outputs):
            lhs = sum(lambda_vec[j] * self.outputs[j, r] for j in range(self.n_dmus))
            output_slack[r] = lhs - self.outputs[k, r]

        return theta, lambda_vec, input_slack, output_slack

    def fit(self):
        self.efficiency_scores = np.zeros(self.n_dmus)
        self.lambdas = []
        self.slack_input = []
        self.slack_output = []

        print(f"\nRunning DEA: {self.returns} - {self.orientation.capitalize()} Oriented")
        print(f"DMUs: {self.n_dmus}, Inputs: {self.n_inputs}, Outputs: {self.n_outputs}\n")

        for k in range(self.n_dmus):
            print(f"Solving DMU {k+1}/{self.n_dmus}...", end='\r')
            score, lambda_vec, slack_in, slack_out = self._solve_input_oriented(k)
            self.efficiency_scores[k] = score
            self.lambdas.append(lambda_vec)
            self.slack_input.append(slack_in)
            self.slack_output.append(slack_out)

        print("\nDEA Complete!\n")
        return self.efficiency_scores


# ============================================================================
# PART 2: DATA LOADING AND PREPARATION
# ============================================================================

print("="*70)
print("FINANCIAL PROTECTION EFFICIENCY (FPE) - DEA ANALYSIS")
print("Model: VRS Input-Oriented")
print("Inputs: CHE_PPP, OOP_pct | Output: FPE")
print("="*70)

# ----------------------------------------------------------------------------
# 2.1: Load Health Expenditure per Capita (CHE per capita PPP)
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("1. Loading: CHE per capita PPP (World Bank Data)")
print("-"*70)

che_file = '/Users/aishaanibajaj/Downloads/CHE_perCapita_PPP_cleaned.csv'

try:
    df_che = pd.read_csv(che_file)
    print(f"✓ Loaded: {che_file}")
    print(f"  Rows: {len(df_che)}, Columns: {len(df_che.columns)}")

    # Reshape from wide to long format
    year_cols = [col for col in df_che.columns if col.isdigit()]

    df_che_long = pd.melt(
        df_che,
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'has_data'],
        value_vars=year_cols,
        var_name='Year',
        value_name='CHE_PPP'
    )

    df_che_long['Year'] = df_che_long['Year'].astype(int)
    df_che_long = df_che_long.dropna(subset=['CHE_PPP'])

    print(f"  Reshaped to: {len(df_che_long)} observations")
    print(f"  Countries: {df_che_long['Country Code'].nunique()}")
    print(f"  Year range: {df_che_long['Year'].min()} - {df_che_long['Year'].max()}")

except FileNotFoundError:
    print(f"✗ File not found: {che_file}")
    print("  Please check the file path and try again.")
    exit()

# ----------------------------------------------------------------------------
# 2.2: Load CHE10 Data (Catastrophic Health Expenditure at 10% threshold)
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("2. Loading: CHE10 (Catastrophic Health Expenditure >10% threshold)")
print("-"*70)

che10_file = '/Users/aishaanibajaj/Downloads/CHE10_cleaned_v2.csv'

try:
    df_che10 = pd.read_csv(che10_file)
    print(f"✓ Loaded: {che10_file}")
    print(f"  Rows: {len(df_che10)}, Columns: {len(df_che10.columns)}")

    # Reshape from wide to long format
    year_cols = [col for col in df_che10.columns if col.isdigit()]

    df_che10_long = pd.melt(
        df_che10,
        id_vars=['Country Name', 'Country Code', 'has_data'],
        value_vars=year_cols,
        var_name='Year',
        value_name='CHE10'
    )

    df_che10_long['Year'] = df_che10_long['Year'].astype(int)
    df_che10_long = df_che10_long.dropna(subset=['CHE10'])

    # Calculate FPE = 100 - CHE10
    df_che10_long['FPE'] = 100 - df_che10_long['CHE10']

    print(f"  Reshaped to: {len(df_che10_long)} observations")
    print(f"  Countries: {df_che10_long['Country Code'].nunique()}")
    print(f"  Year range: {df_che10_long['Year'].min()} - {df_che10_long['Year'].max()}")
    print(f"  FPE range: {df_che10_long['FPE'].min():.1f} - {df_che10_long['FPE'].max():.1f}")

except FileNotFoundError:
    print(f"✗ File not found: {che10_file}")
    print("  Please check the file path and try again.")
    exit()

# ----------------------------------------------------------------------------
# 2.3: Load OOP (Out-of-pocket expenditure as % of current health expenditure)
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("3. Loading: OOP (Out-of-pocket expenditure as % of current health expenditure)")
print("-"*70)

oop_file = '/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /OOP_cleaned.csv'  # Using the file you provided

try:
    df_oop = pd.read_csv(oop_file)
    print(f"✓ Loaded: {oop_file}")
    print(f"  Rows: {len(df_oop)}, Columns: {len(df_oop.columns)}")

    # Reshape from wide to long format
    year_cols = [col for col in df_oop.columns if col.isdigit()]

    df_oop_long = pd.melt(
        df_oop,
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'has_data'],
        value_vars=year_cols,
        var_name='Year',
        value_name='OOP_pct'
    )

    df_oop_long['Year'] = df_oop_long['Year'].astype(int)
    df_oop_long = df_oop_long.dropna(subset=['OOP_pct'])

    print(f"  Reshaped to: {len(df_oop_long)} observations")
    print(f"  Countries: {df_oop_long['Country Code'].nunique()}")
    print(f"  Year range: {df_oop_long['Year'].min()} - {df_oop_long['Year'].max()}")

except FileNotFoundError:
    print(f"✗ File not found: {oop_file}")
    print("  Please check the file path and try again.")
    print("  Make sure OOP_cleaned.csv is in the current directory.")
    exit()

# ----------------------------------------------------------------------------
# 2.4: Merge All Data
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("4. Merging all datasets")
print("-"*70)

# Merge all datasets on Country Code and Year
df_merged = df_che_long[['Country Name', 'Country Code', 'Year', 'CHE_PPP']].copy()

# Merge CHE10/FPE
df_merged = df_merged.merge(
    df_che10_long[['Country Code', 'Year', 'CHE10', 'FPE']],
    on=['Country Code', 'Year'],
    how='inner'
)

# Merge OOP
df_merged = df_merged.merge(
    df_oop_long[['Country Code', 'Year', 'OOP_pct']],
    on=['Country Code', 'Year'],
    how='inner'
)

print(f"✓ Merged dataset: {len(df_merged)} observations")
print(f"  Countries: {df_merged['Country Code'].nunique()}")
print(f"  Year range: {df_merged['Year'].min()} - {df_merged['Year'].max()}")
print(f"\n  Columns: {list(df_merged.columns)}")

# Show sample
print("\n  Sample data (first 10 rows):")
print(df_merged.head(10).to_string(index=False))

# ----------------------------------------------------------------------------
# 2.5: Define DEA Model Specification
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("5. DEA MODEL SPECIFICATION")
print("-"*70)

# Define DEA Inputs and Outputs
# INPUTS (Health Financing Resources)
INPUTS = [
    'CHE_PPP',      # Current health expenditure per capita (PPP)
    'OOP_pct'       # Out-of-pocket expenditure as % of current health expenditure
]

# OUTPUTS (Financial Protection Outcomes)
OUTPUTS = [
    'FPE'           # Financial Protection Efficiency = 100 - CHE10
]

print("\n  INPUTS (Resources):")
for inp in INPUTS:
    print(f"    - {inp}")

print("\n  OUTPUTS (Protection):")
for out in OUTPUTS:
    print(f"    - {out}")

print("\n  Model: VRS (Variable Returns to Scale)")
print("  Orientation: Input-Oriented")
print("  Rationale: Countries control health spending more than health outcomes")

# Check for missing data
print("\n  Checking for missing values...")
for col in INPUTS + OUTPUTS:
    missing = df_merged[col].isnull().sum()
    if missing > 0:
        print(f"    {col}: {missing} missing values")

# Drop rows with missing data
df_clean = df_merged.dropna(subset=INPUTS + OUTPUTS)
print(f"\n  Observations before cleaning: {len(df_merged)}")
print(f"  Observations after cleaning:  {len(df_clean)}")

# ----------------------------------------------------------------------------
# 2.6: Run DEA Analysis
# ----------------------------------------------------------------------------

print("\n" + "="*70)
print("6. RUNNING DEA ANALYSIS")
print("="*70)

# Prepare X (inputs) and Y (outputs)
X = df_clean[INPUTS].values
Y = df_clean[OUTPUTS].values

print(f"\n  DEA Inputs ({len(INPUTS)}):")
for inp in INPUTS:
    print(f"    - {inp}")

print(f"\n  DEA Outputs ({len(OUTPUTS)}):")
for out in OUTPUTS:
    print(f"    - {out}")

print(f"\n  DMUs: {len(df_clean)} (country-year observations)")

# Choose model specification
dea = DEA(
    inputs=X,
    outputs=Y,
    returns='VRS',      # Variable Returns to Scale
    orientation='input' # Input-oriented
)

# Fit the model
scores = dea.fit()

# ----------------------------------------------------------------------------
# 2.7: Results
# ----------------------------------------------------------------------------

print("\n" + "="*70)
print("7. DEA RESULTS")
print("="*70)

# Add scores to the dataframe
df_results = df_clean.copy()
df_results['DEA_Efficiency'] = scores

# Summary statistics
print(f"\n  SUMMARY STATISTICS")
print(f"  {'='*50}")
print(f"  Total Observations:   {len(scores)}")
print(f"  Mean Efficiency:      {scores.mean():.4f}")
print(f"  Std Deviation:        {scores.std():.4f}")
print(f"  Minimum Score:        {scores.min():.4f}")
print(f"  Maximum Score:        {scores.max():.4f}")
print(f"  Efficient DMUs:       {sum(scores == 1.0)} out of {len(scores)}")
print(f"  Efficient Percentage: {sum(scores == 1.0)/len(scores)*100:.1f}%")

# Show top 10 most efficient DMUs
print(f"\n  TOP 10 MOST EFFICIENT COUNTRIES-YEARS:")
print(f"  {'Country':<30} {'Year':<6} {'Efficiency':<12} {'FPE':<8} {'CHE_PPP':<12} {'OOP_pct':<10}")
print(f"  {'-'*80}")
top_10 = df_results.nlargest(10, 'DEA_Efficiency')
for _, row in top_10.iterrows():
    print(f"  {row['Country Name'][:28]:<30} {row['Year']:<6} {row['DEA_Efficiency']:<12.4f} {row['FPE']:<8.1f} {row['CHE_PPP']:<12.0f} {row['OOP_pct']:<10.2f}")

# Show bottom 10 least efficient DMUs
print(f"\n  BOTTOM 10 LEAST EFFICIENT COUNTRIES-YEARS:")
print(f"  {'Country':<30} {'Year':<6} {'Efficiency':<12} {'FPE':<8} {'CHE_PPP':<12} {'OOP_pct':<10}")
print(f"  {'-'*80}")
bottom_10 = df_results.nsmallest(10, 'DEA_Efficiency')
for _, row in bottom_10.iterrows():
    print(f"  {row['Country Name'][:28]:<30} {row['Year']:<6} {row['DEA_Efficiency']:<12.4f} {row['FPE']:<8.1f} {row['CHE_PPP']:<12.0f} {row['OOP_pct']:<10.2f}")

# Efficient countries by year
print(f"\n  EFFICIENT COUNTRIES BY YEAR:")
efficient = df_results[df_results['DEA_Efficiency'] == 1.0]
if len(efficient) > 0:
    for year in sorted(df_results['Year'].unique()):
        year_efficient = efficient[efficient['Year'] == year]
        if len(year_efficient) > 0:
            countries = ', '.join(year_efficient['Country Name'].tolist())
            print(f"    {year}: {countries}")
else:
    print("    No countries achieved full efficiency")

# ----------------------------------------------------------------------------
# 2.8: Save Results
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("8. Saving Results")
print("-"*70)

# Save full results
output_file = 'dea_results_fpe_oop.csv'
df_results.to_csv(output_file, index=False)
print(f"✓ Full results saved to: {output_file}")

# Save just efficiency scores for Stage 2
stage2_file = 'efficiency_scores_for_stage2_oop.csv'
df_stage2 = df_results[['Country Name', 'Country Code', 'Year', 'DEA_Efficiency', 'FPE', 'CHE_PPP', 'OOP_pct']].copy()
df_stage2.to_csv(stage2_file, index=False)
print(f"✓ Stage 2 data saved to: {stage2_file}")

# Save summary statistics
summary_file = 'dea_summary_stats_oop.csv'
summary_df = pd.DataFrame({
    'Metric': ['Observations', 'Countries', 'Years', 'Mean_Score', 'Std_Dev', 'Min_Score', 'Max_Score', 'Efficient_DMUs', 'Efficient_Percentage'],
    'Value': [
        len(df_results),
        df_results['Country Code'].nunique(),
        df_results['Year'].nunique(),
        scores.mean(),
        scores.std(),
        scores.min(),
        scores.max(),
        sum(scores == 1.0),
        f"{sum(scores == 1.0)/len(scores)*100:.1f}%"
    ]
})
summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary statistics saved to: {summary_file}")

# ----------------------------------------------------------------------------
# 2.9: Visualizations
# ----------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Histogram of efficiency scores
    axes[0, 0].hist(scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Efficient Frontier')
    axes[0, 0].axvline(scores.mean(), color='green', linestyle='-', linewidth=2,
                      label=f'Mean: {scores.mean():.3f}')
    axes[0, 0].set_xlabel('DEA Efficiency Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of FPE Efficiency Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. FPE vs Efficiency
    axes[0, 1].scatter(df_results['FPE'], scores, alpha=0.6, s=30, c='steelblue')
    axes[0, 1].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Efficient')
    axes[0, 1].set_xlabel('FPE (100 - CHE10)')
    axes[0, 1].set_ylabel('DEA Efficiency Score')
    axes[0, 1].set_title('FPE vs DEA Efficiency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. CHE_PPP vs Efficiency
    axes[0, 2].scatter(df_results['CHE_PPP'], scores, alpha=0.6, s=30, c='steelblue')
    axes[0, 2].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Efficient')
    axes[0, 2].set_xlabel('CHE per capita PPP (current international $)')
    axes[0, 2].set_ylabel('DEA Efficiency Score')
    axes[0, 2].set_title('Health Expenditure vs Efficiency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. OOP_pct vs Efficiency
    axes[1, 0].scatter(df_results['OOP_pct'], scores, alpha=0.6, s=30, c='steelblue')
    axes[1, 0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Efficient')
    axes[1, 0].set_xlabel('OOP (% of Current Health Expenditure)')
    axes[1, 0].set_ylabel('DEA Efficiency Score')
    axes[1, 0].set_title('Out-of-Pocket Spending vs Efficiency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Efficiency by Year (boxplot)
    years = sorted(df_results['Year'].unique())
    year_data = [df_results[df_results['Year'] == y]['DEA_Efficiency'].values for y in years]
    bp = axes[1, 1].boxplot(year_data, labels=years, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[1, 1].axhline(1.0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('DEA Efficiency Score')
    axes[1, 1].set_title('Efficiency Scores by Year')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Input-Output Relationship (3D-like scatter)
    # Plot CHE_PPP vs OOP_pct, colored by efficiency
    scatter = axes[1, 2].scatter(df_results['CHE_PPP'], df_results['OOP_pct'],
                                c=scores, cmap='RdYlGn', s=30, alpha=0.7)
    axes[1, 2].set_xlabel('CHE per capita PPP')
    axes[1, 2].set_ylabel('OOP (% of Current Health Expenditure)')
    axes[1, 2].set_title('Input Space Colored by Efficiency')
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label('Efficiency Score')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dea_visualization_oop.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: dea_visualization_oop.png")
    plt.show()

except ImportError:
    print("\n  (Visualization skipped - matplotlib not installed)")

print("\n" + "="*70)
print("✓ STAGE 1 DEA ANALYSIS COMPLETE!")
print("="*70)
print("\n  Model Used:")
print("    - Inputs:  CHE_PPP, OOP_pct")
print("    - Output:  FPE (100 - CHE10)")
print("    - Returns: VRS (Variable Returns to Scale)")
print("    - Orientation: Input-Oriented")
print("\n  Next Step:")
print("    Use 'efficiency_scores_for_stage2_oop.csv' for Stage 2")
print("    (Simar-Wilson Double-Bootstrap Truncated Regression)")
print("="*70)