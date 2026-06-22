"""
Equation 1 Regression: CHE10 ~ OOP_share + GGHE_D_share
Panel data, three estimators: Pooled OLS, Fixed Effects, Random Effects

CHE10  = Population with health spending >10% of household budget (%)
OOP    = Out-of-pocket expenditure (% of current health expenditure)
GGHE_D = Domestic general government health expenditure (% of general govt expenditure)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
from linearmodels.panel import compare
import statsmodels.api as sm

# ── 1. Load & melt to long format ────────────────────────────────────────────

YEAR_COLS = [str(y) for y in range(2000, 2025)]

def load_and_melt(path, value_name):
    df = pd.read_csv(path)
    df = df[df["has_data"] == True].copy()
    # Drop aggregate regions — keep only ISO 3-letter country codes
    df = df[df["Country Code"].str.len() == 3]
    df = df.melt(
        id_vars=["Country Code", "Country Name"],
        value_vars=[c for c in YEAR_COLS if c in df.columns],
        var_name="year",
        value_name=value_name,
    )
    df["year"] = df["year"].astype(int)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df[["Country Code", "Country Name", "year", value_name]]


che  = load_and_melt("/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /CHE10_cleaned.csv",  "CHE10")
oop  = load_and_melt("/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /OOP_cleaned.csv",    "OOP")
gghe = load_and_melt("/Users/aishaanibajaj/Downloads/GGHE_D_cleaned.csv", "GGHE_D")

# ── 2. Merge on country + year ────────────────────────────────────────────────

df = (
    che
    .merge(oop[["Country Code", "year", "OOP"]],     on=["Country Code", "year"], how="left")
    .merge(gghe[["Country Code", "year", "GGHE_D"]], on=["Country Code", "year"], how="left")
)

# ── 3. Drop rows missing any variable ────────────────────────────────────────

df_clean = df.dropna(subset=["CHE10", "OOP", "GGHE_D"]).copy()

print("=" * 60)
print("DATA SUMMARY AFTER MERGE & LISTWISE DELETION")
print("=" * 60)
print(f"Observations         : {len(df_clean)}")
print(f"Unique countries     : {df_clean['Country Code'].nunique()}")
print(f"Years represented    : {sorted(df_clean['year'].unique())}")
print(f"\nVariable means:")
print(df_clean[["CHE10","OOP","GGHE_D"]].describe().round(2))

# ── 4. Set MultiIndex for linearmodels ───────────────────────────────────────

df_panel = df_clean.set_index(["Country Code", "year"])

Y = df_panel["CHE10"]
X = sm.add_constant(df_panel[["OOP", "GGHE_D"]])

# ── 5. Run three estimators ───────────────────────────────────────────────────

# Pooled OLS
pooled = PooledOLS(Y, X).fit(cov_type="clustered", cluster_entity=True)

# Fixed Effects (two-way: entity + time)
X_fe = df_panel[["OOP", "GGHE_D"]]  # FE absorbs intercept
fe   = PanelOLS(Y, X_fe, entity_effects=True, time_effects=True).fit(
           cov_type="clustered", cluster_entity=True)

# Random Effects
re   = RandomEffects(Y, X).fit(cov_type="robust")

# ── 6. Print results ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EQUATION 1 RESULTS")
print("CHE10 = β0 + β1·OOP + β2·GGHE_D + ε")
print("=" * 60)

for label, res in [("Pooled OLS", pooled), ("Fixed Effects (two-way)", fe), ("Random Effects", re)]:
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    params = res.params
    pvals  = res.pvalues
    ses    = res.std_errors
    ci     = res.conf_int()

    for var in params.index:
        stars = (
            "***" if pvals[var] < 0.01 else
            "**"  if pvals[var] < 0.05 else
            "*"   if pvals[var] < 0.10 else ""
        )
        print(f"  {var:<12}  coef={params[var]:+8.4f}  SE={ses[var]:.4f}  "
              f"p={pvals[var]:.4f}  {stars}")

    try:
        r2 = res.rsquared
    except Exception:
        r2 = np.nan
    print(f"\n  R²={r2:.4f}   N={int(res.nobs)}")

# ── 7. Hausman-style comparison (FE vs RE) — eigenvalue check ────────────────

print("\n" + "=" * 60)
print("MODEL COMPARISON (linearmodels)")
print("=" * 60)
comp = compare({"Pooled OLS": pooled, "Fixed Effects": fe, "Random Effects": re})
print(comp.summary)

# ── 8. Save residuals from the preferred FE model ────────────────────────────

residuals = fe.resids.rename("residuals_eq1")
residuals_df = residuals.reset_index()
residuals_df = residuals_df.merge(
    df_clean[["Country Code", "Country Name", "year"]],
    on=["Country Code", "year"], how="left"
)
import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq1_residuals.csv")
residuals_df.to_csv(output_path, index=False)
print(f"\nResiduals from Fixed Effects model saved → {output_path}")

# ── 9. Residual diagnostics ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESIDUAL DIAGNOSTICS (Fixed Effects)")
print("=" * 60)
r = residuals.values
print(f"  Mean    : {r.mean():.6f}  (should be ~0)")
print(f"  Std Dev : {r.std():.4f}")
print(f"  Min/Max : {r.min():.4f} / {r.max():.4f}")

# Countries with largest absolute residuals
residuals_df["abs_resid"] = residuals_df["residuals_eq1"].abs()
top10 = residuals_df.nlargest(10, "abs_resid")[
    ["Country Name", "year", "residuals_eq1"]
].reset_index(drop=True)
print(f"\n  Top 10 observations by |residual|:")
print(top10.to_string(index=False))