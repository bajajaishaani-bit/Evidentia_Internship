import pandas as pd
import numpy as np

# ============================================================
# EQUATION 2 DATASET BUILDER
# Run this in PyCharm from your Internship project folder
# Requires: panel_data_full.csv (from Pooled OLS.py)
#           wgi_country_means.csv
#           gini_country_means.csv
# ============================================================

PANEL_PATH  = "panel_data_full.csv"
WGI_PATH    = "/Users/aishaanibajaj/Downloads/wgi_country_means.csv"
GINI_PATH   = "/Users/aishaanibajaj/Downloads/gini_country_means.csv"
OUTPUT_PATH = "eq2_dataset.csv"
MIN_OBS     = 2

# ── 1. Load panel residuals and compute country means ────────────────────────
panel = pd.read_csv(PANEL_PATH)
print(f"Panel loaded: {len(panel)} obs, {panel['Country Name'].nunique()} countries")

# Country code mapping — needed to merge with GINI (which uses ISO codes)
# First try using a Country Code column if it exists
if 'Country Code' not in panel.columns:
    # Build from WGI lookup
    wgi_lookup = pd.read_csv(WGI_PATH)[["Country Name", "Country Code"]].drop_duplicates()
    # Manual overrides for WHO vs World Bank name differences
    overrides = {
        "Czechia": "CZE", "Korea, Rep.": "KOR", "Iran, Islamic Rep.": "IRN",
        "Egypt, Arab Rep.": "EGY", "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG",
        "Yemen, Rep.": "YEM", "Gambia, The": "GMB", "Lao PDR": "LAO",
        "West Bank and Gaza": "PSE", "Viet Nam": "VNM", "Syrian Arab Republic": "SYR",
        "Russian Federation": "RUS", "Slovak Republic": "SVK",
        "St. Kitts and Nevis": "KNA", "St. Lucia": "LCA", "Cabo Verde": "CPV",
        "Eswatini": "SWZ", "Timor-Leste": "TLS", "North Macedonia": "MKD",
    }
    name_map = dict(zip(wgi_lookup["Country Name"], wgi_lookup["Country Code"]))
    name_map.update(overrides)
    panel["Country Code"] = panel["Country Name"].map(name_map)

# Aggregate to country level
country_res = panel.groupby(["Country Name", "Country Code"]).agg(
    n_obs        = ("residuals", "count"),
    mean_residual= ("residuals", "mean"),
    mean_che     = ("che", "mean"),
    mean_oop     = ("oop_share", "mean"),
    mean_gov     = ("gov_share", "mean"),
    year_min     = ("year", "min"),
    year_max     = ("year", "max"),
).reset_index()

# Filter to countries with sufficient data
country_res = country_res[country_res["n_obs"] >= MIN_OBS].copy()
print(f"Countries with n_obs >= {MIN_OBS}: {len(country_res)}")

# ── 2. Merge WGI and GINI ────────────────────────────────────────────────────
wgi  = pd.read_csv(WGI_PATH)
gini = pd.read_csv(GINI_PATH)

df = country_res.merge(wgi,  on="Country Code", how="left", suffixes=("", "_wgi"))
df = df.merge(gini, on="Country Code", how="left")

print(f"\nAfter merge:")
print(f"  WGI missing:  {df['WGI_composite'].isna().sum()} countries")
print(f"  GINI missing: {df['GINI_mean'].isna().sum()} countries")

# Drop countries missing either key variable
df_clean = df.dropna(subset=["WGI_composite", "GINI_mean"]).copy()
print(f"  Complete cases: {len(df_clean)} countries")

# ── 3. Save ──────────────────────────────────────────────────────────────────
cols_to_keep = [
    "Country Name", "Country Code", "n_obs",
    "mean_residual", "mean_che", "mean_oop", "mean_gov",
    "year_min", "year_max",
    "WGI_composite", "WGI_GE", "WGI_CC", "WGI_RL",
    "GINI_mean"
]
df_clean[cols_to_keep].to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {OUTPUT_PATH}")

# ── 4. Preview ───────────────────────────────────────────────────────────────
print("\n--- Top 10 underperformers (actual CHE > predicted) ---")
print(df_clean.nlargest(10, "mean_residual")[
    ["Country Name", "n_obs", "mean_residual", "WGI_composite", "GINI_mean"]
].to_string(index=False))

print("\n--- Top 10 overperformers (actual CHE < predicted) ---")
print(df_clean.nsmallest(10, "mean_residual")[
    ["Country Name", "n_obs", "mean_residual", "WGI_composite", "GINI_mean"]
].to_string(index=False))

print("\n--- Descriptive statistics ---")
print(df_clean[["mean_residual", "WGI_composite", "GINI_mean"]].describe().round(3))