"""
Hausman Test: Fixed Effects vs Random Effects
For Equation 1: CHE10 ~ OOP + GGHE_D

H0: RE is consistent and efficient (use Random Effects)
H1: FE is consistent, RE is not (use Fixed Effects)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os

from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm
from scipy import stats

# ── 1. Load & prep (same as eq1_regression.py) ───────────────────────────────

YEAR_COLS = [str(y) for y in range(2000, 2025)]

def load_and_melt(path, value_name):
    df = pd.read_csv(path)
    df = df[df["has_data"] == True].copy()
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


BASE = os.path.dirname(os.path.abspath(__file__))

che  = load_and_melt(os.path.join(BASE, "/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /CHE10_cleaned.csv"),  "CHE10")
oop  = load_and_melt(os.path.join(BASE, "/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /OOP_cleaned.csv"),    "OOP")
gghe = load_and_melt(os.path.join(BASE, "/Users/aishaanibajaj/Downloads/GGHE_D_cleaned.csv"), "GGHE_D")

df = (
    che
    .merge(oop[["Country Code", "year", "OOP"]],     on=["Country Code", "year"], how="left")
    .merge(gghe[["Country Code", "year", "GGHE_D"]], on=["Country Code", "year"], how="left")
)

df_clean = df.dropna(subset=["CHE10", "OOP", "GGHE_D"]).copy()
df_panel  = df_clean.set_index(["Country Code", "year"])

Y = df_panel["CHE10"]
X_vars = ["OOP", "GGHE_D"]

# ── 2. Fit FE and RE (entity effects only — Hausman tests entity assumption) ─
#    Note: two-way FE absorbs time effects too; for the Hausman test we use
#    one-way entity FE vs RE, which is the standard formulation.

X_fe = df_panel[X_vars]
X_re = sm.add_constant(df_panel[X_vars])

fe = PanelOLS(Y, X_fe, entity_effects=True).fit(cov_type="unadjusted")
re = RandomEffects(Y, X_re).fit(cov_type="unadjusted")

# ── 3. Hausman test (manual chi-squared) ─────────────────────────────────────
#
#    H = (b_FE - b_RE)' [Var(b_FE) - Var(b_RE)]^{-1} (b_FE - b_RE)
#    Under H0 ~ χ²(k), k = number of regressors (2 here)

b_fe = fe.params[X_vars].values
b_re = re.params[X_vars].values   # excludes const

V_fe = fe.cov.loc[X_vars, X_vars].values
V_re = re.cov.loc[X_vars, X_vars].values

diff     = b_fe - b_re
V_diff   = V_fe - V_re

# Check if V_diff is positive semi-definite (can fail with small samples)
eigenvalues = np.linalg.eigvalsh(V_diff)

print("=" * 60)
print("HAUSMAN TEST: Fixed Effects vs Random Effects")
print("H0: RE is consistent (individual effects uncorrelated with X)")
print("H1: FE is consistent, RE is not")
print("=" * 60)

print(f"\nCoefficient comparison:")
print(f"{'Variable':<12} {'FE':>10} {'RE':>10} {'Difference':>12}")
print(f"{'─'*46}")
for i, v in enumerate(X_vars):
    print(f"{v:<12} {b_fe[i]:>10.4f} {b_re[i]:>10.4f} {diff[i]:>12.4f}")

print(f"\nV(b_FE) - V(b_RE) eigenvalues: {eigenvalues.round(6)}")

if np.all(eigenvalues > -1e-8):  # allow tiny numerical noise
    H_stat = float(diff @ np.linalg.inv(V_diff) @ diff)
    df_chi  = len(X_vars)
    p_value = 1 - stats.chi2.cdf(H_stat, df=df_chi)

    print(f"\n{'─'*46}")
    print(f"  Hausman statistic (χ²) : {H_stat:.4f}")
    print(f"  Degrees of freedom     : {df_chi}")
    print(f"  p-value                : {p_value:.4f}")
    print(f"{'─'*46}")

    if p_value < 0.05:
        print("\n  ✗ Reject H0 (p < 0.05)")
        print("  → Use FIXED EFFECTS: individual effects are correlated with regressors.")
    else:
        print("\n  ✓ Fail to reject H0 (p ≥ 0.05)")
        print("  → Use RANDOM EFFECTS: RE is consistent and more efficient.")

else:
    # V_diff not PSD — common with sparse/unbalanced panels
    # Fall back to Wooldridge's robust version via artificial regression
    print("\n  ⚠ V(b_FE) - V(b_RE) is not positive semi-definite.")
    print("  Standard Hausman test unreliable. Running artificial regression test instead.")
    print()

    # ── Mundlak (1978) artificial regression ─────────────────────────────────
    # Add group means of regressors to RE model.
    # Significant joint test on means ⟹ reject RE.

    df_aug = df_clean.copy()
    for v in X_vars:
        df_aug[f"{v}_mean"] = df_aug.groupby("Country Code")[v].transform("mean")

    df_aug_panel = df_aug.set_index(["Country Code", "year"])
    mean_vars    = [f"{v}_mean" for v in X_vars]
    X_mundlak    = sm.add_constant(df_aug_panel[X_vars + mean_vars])

    mundlak = RandomEffects(df_aug_panel["CHE10"], X_mundlak).fit(cov_type="robust")

    # Joint F/Wald test on the mean terms
    from linearmodels.panel.results import compare
    restriction = np.zeros((len(mean_vars), len(mundlak.params)))
    for i, mv in enumerate(mean_vars):
        idx = list(mundlak.params.index).index(mv)
        restriction[i, idx] = 1.0

    wald_stat = float(
        mundlak.params[mean_vars].values
        @ np.linalg.inv(mundlak.cov.loc[mean_vars, mean_vars].values)
        @ mundlak.params[mean_vars].values
    )
    df_wald = len(mean_vars)
    p_wald  = 1 - stats.chi2.cdf(wald_stat, df=df_wald)

    print("  Mundlak (1978) auxiliary regression — group means added to RE model:")
    print(f"  {'Variable':<14} {'Coef':>8} {'p-value':>10}")
    print(f"  {'─'*36}")
    for mv in mean_vars:
        print(f"  {mv:<14} {mundlak.params[mv]:>8.4f}  {mundlak.pvalues[mv]:>8.4f}")

    print(f"\n  Joint Wald statistic (χ²) : {wald_stat:.4f}")
    print(f"  Degrees of freedom        : {df_wald}")
    print(f"  p-value                   : {p_wald:.4f}")

    if p_wald < 0.05:
        print("\n  ✗ Reject H0 (p < 0.05)")
        print("  → Use FIXED EFFECTS.")
    else:
        print("\n  ✓ Fail to reject H0 (p ≥ 0.05)")
        print("  → Use RANDOM EFFECTS.")

print("\n" + "=" * 60)
print("NOTE: This test uses entity effects only (one-way FE vs RE).")
print("Time effects are a separate modelling choice tested via F-test")
print("on time dummies, not the Hausman framework.")
print("=" * 60)