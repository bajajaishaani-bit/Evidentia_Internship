import pandas as pd
import numpy as np

# ============================================================
# EQUATION 2 DATASET BUILDER WITH REGION CONTROLS
# ============================================================

PANEL_PATH  = "panel_data_full.csv"
WGI_PATH    = "/Users/aishaanibajaj/Downloads/wgi_country_means.csv"
GINI_PATH   = "/Users/aishaanibajaj/Downloads/gini_country_means.csv"
WB_CLASS_PATH = "/Users/aishaanibajaj/Downloads/CLASS_2025_10_07.xlsx"
OUTPUT_PATH = "eq2_dataset_with_regions.csv"
MIN_OBS     = 2

# ── 1. Load World Bank country classifications ──────────────────────────────
wb_class = pd.read_excel(WB_CLASS_PATH, sheet_name="List of economies")
print(f"WB classification loaded: {len(wb_class)} economies")

# Keep only relevant columns and rename for clarity
wb_class = wb_class[['Economy', 'Code', 'Region']].rename(columns={
    'Economy': 'Country_Name_WB',
    'Code': 'Country Code',
    'Region': 'WB_Region'
})

# ── 2. Load panel residuals and compute country means ────────────────────────
panel = pd.read_csv(PANEL_PATH)
print(f"Panel loaded: {len(panel)} obs, {panel['Country Name'].nunique()} countries")

# Country code mapping
if 'Country Code' not in panel.columns:
    wgi_lookup = pd.read_csv(WGI_PATH)[["Country Name", "Country Code"]].drop_duplicates()
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

# ── 3. Merge region information ──────────────────────────────────────────────
# Merge region data with country_res
df = country_res.merge(wb_class[['Country Code', 'WB_Region']],
                       on='Country Code',
                       how='left')

print(f"\nRegion merge results:")
print(f"  Countries with region info: {df['WB_Region'].notna().sum()}")
print(f"  Countries missing region: {df['WB_Region'].isna().sum()}")

# ── 4. Merge WGI and GINI ────────────────────────────────────────────────────
wgi  = pd.read_csv(WGI_PATH)
gini = pd.read_csv(GINI_PATH)

df = df.merge(wgi,  on="Country Code", how="left", suffixes=("", "_wgi"))
df = df.merge(gini, on="Country Code", how="left")

print(f"\nAfter merge:")
print(f"  WGI missing:  {df['WGI_composite'].isna().sum()} countries")
print(f"  GINI missing: {df['GINI_mean'].isna().sum()} countries")

# Drop countries missing key variables
df_clean = df.dropna(subset=["WGI_composite", "GINI_mean"]).copy()
print(f"  Complete cases: {len(df_clean)} countries")

# ── 5. Create region dummies (for regression) ──────────────────────────────
# Get unique regions (excluding NaN)
unique_regions = df_clean['WB_Region'].dropna().unique()
print(f"\nRegions in dataset: {sorted(unique_regions)}")

# Create dummy variables for each region
region_dummies = pd.get_dummies(df_clean['WB_Region'], prefix='Region')

# Option 1: Drop one region as base (e.g., 'Sub-Saharan Africa')
# This creates k-1 dummies to avoid perfect multicollinearity
# If you want to keep all dummies, use drop_first=False and add a constant in regression
region_dummies_dropped = region_dummies.drop(columns=['Region_Sub-Saharan Africa'], errors='ignore')

# Concatenate with the main dataframe
df_final = pd.concat([df_clean, region_dummies_dropped], axis=1)

# ── 6. Save ──────────────────────────────────────────────────────────────────
cols_to_keep = [
    "Country Name", "Country Code", "n_obs",
    "mean_residual", "mean_che", "mean_oop", "mean_gov",
    "year_min", "year_max",
    "WGI_composite", "WGI_GE", "WGI_CC", "WGI_RL",
    "GINI_mean", "WB_Region"
] + list(region_dummies_dropped.columns)

df_final[cols_to_keep].to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {OUTPUT_PATH}")

# ── 7. Preview ───────────────────────────────────────────────────────────────
print("\n--- Sample with region info ---")
print(df_final[["Country Name", "WB_Region", "mean_residual", "WGI_composite", "GINI_mean"]].head(10).to_string(index=False))

print("\n--- Region distribution ---")
print(df_final['WB_Region'].value_counts())

print("\n--- Descriptive statistics by region ---")
print(df_final.groupby('WB_Region')[['mean_residual', 'WGI_composite', 'GINI_mean']].mean().round(3))