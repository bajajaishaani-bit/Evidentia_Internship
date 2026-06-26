import pandas as pd
import numpy as np

# ============================================================
# EQUATION 2 DATASET BUILDER WITH HDI CONTROLS
# ============================================================

PANEL_PATH  = "panel_data_full.csv"
WGI_PATH    = "/Users/aishaanibajaj/Downloads/wgi_country_means.csv"
GINI_PATH   = "/Users/aishaanibajaj/Downloads/gini_country_means.csv"
HDI_PATH    = "/Users/aishaanibajaj/Downloads/HDR25_Statistical_Annex_HDI_Table.xlsx"
OUTPUT_PATH = "eq2_dataset_with_hdi.csv"
MIN_OBS     = 2

# ── 1. Load HDI data ──────────────────────────────────────────────────────────
print("Loading HDI data...")
hdi_data = pd.read_excel(HDI_PATH, sheet_name="Table 1. HDI", header=None)
print(f"Raw HDI data shape: {hdi_data.shape}")

# We can see from the output that the data starts at row 7 (with "Very high human development")
# And the country data starts at row 8
# So we'll read from row 7, and use row 7 as header, then skip to country data

# Read the data with header at row 7 (where "Very high human development" is)
hdi_data = pd.read_excel(HDI_PATH, sheet_name="Table 1. HDI", header=7)
print(f"\nHDI data shape after header: {hdi_data.shape}")
print(f"Columns: {hdi_data.columns.tolist()}")

# Now we need to rename columns properly
# Based on the structure we saw:
# Column 0: HDI Rank (but has "Very high human development" in first row)
# Column 1: Country
# Column 2: HDI Value
# Column 3: (empty/life expectancy label)
# Column 4: Life Expectancy
# Column 5: (empty/expected schooling label)
# Column 6: Expected Schooling
# Column 7: (empty/mean schooling label)
# Column 8: Mean Schooling
# Column 9: (empty/GNI label)
# Column 10: GNI per capita
# Column 11: (empty/GNI rank diff label)
# Column 12: GNI rank minus HDI rank
# Column 13: (empty)
# Column 14: HDI rank 2022

# Let's rename the columns
hdi_data.columns = ['HDI_Rank_Label', 'Country', 'HDI_Value', 'Life_Expectancy_Label',
                    'Life_Expectancy', 'Expected_Schooling_Label', 'Expected_Schooling',
                    'Mean_Schooling_Label', 'Mean_Schooling', 'GNI_Label',
                    'GNI_per_capita', 'GNI_Rank_Diff_Label', 'GNI_HDI_Rank_Diff',
                    'Extra_Column', 'HDI_Rank_2022']

print(f"\nRenamed columns: {hdi_data.columns.tolist()}")

# ── 2. Clean the data ──────────────────────────────────────────────────────────
# Remove rows that are not countries (like "Very high human development", "High human development", etc.)
country_patterns = ['Very high human development', 'High human development', 'Medium human development',
                    'Low human development', 'Developing countries', 'Regions', 'Arab States',
                    'East Asia and the Pacific', 'Europe and Central Asia', 'Latin America and the Caribbean',
                    'South Asia', 'Sub-Saharan Africa', 'Least developed countries',
                    'Small island developing states', 'Organisation for Economic Co-operation and Development',
                    'World', 'Notes', 'Definitions', 'Main data sources', 'Other countries or territories',
                    'Human development groups', 'Organisation', 'Organisation for Economic']

# Filter to keep only countries
hdi_data = hdi_data[~hdi_data['Country'].astype(str).str.strip().isin(country_patterns)]
hdi_data = hdi_data[~hdi_data['Country'].astype(str).str.contains('Human development|Regions|countries|OECD|World|Notes|Definitions|---', na=False)]
hdi_data = hdi_data[hdi_data['HDI_Value'].notna()]

# Convert to numeric
hdi_data['HDI_Value'] = pd.to_numeric(hdi_data['HDI_Value'], errors='coerce')
hdi_data['Life_Expectancy'] = pd.to_numeric(hdi_data['Life_Expectancy'], errors='coerce')
hdi_data['Expected_Schooling'] = pd.to_numeric(hdi_data['Expected_Schooling'], errors='coerce')
hdi_data['Mean_Schooling'] = pd.to_numeric(hdi_data['Mean_Schooling'], errors='coerce')
hdi_data['GNI_per_capita'] = pd.to_numeric(hdi_data['GNI_per_capita'], errors='coerce')
hdi_data['GNI_HDI_Rank_Diff'] = pd.to_numeric(hdi_data['GNI_HDI_Rank_Diff'], errors='coerce')
hdi_data['HDI_Rank_2022'] = pd.to_numeric(hdi_data['HDI_Rank_2022'], errors='coerce')

# Also convert HDI_Rank_Label to numeric (it has the rank numbers)
hdi_data['HDI_Rank'] = pd.to_numeric(hdi_data['HDI_Rank_Label'], errors='coerce')

print(f"\nCleaned HDI data: {len(hdi_data)} countries")
print(f"HDI value range: {hdi_data['HDI_Value'].min():.3f} - {hdi_data['HDI_Value'].max():.3f}")
print(f"Countries with valid HDI: {hdi_data['HDI_Value'].notna().sum()}")

# ── 3. Load panel residuals and compute country means ────────────────────────
panel = pd.read_csv(PANEL_PATH)
print(f"\nPanel loaded: {len(panel)} obs, {panel['Country Name'].nunique()} countries")

# Country code mapping for panel data
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

# ── 4. Merge HDI data ──────────────────────────────────────────────────────
# Create a mapping for country name variations between datasets
hdi_country_mapping = {
    'United States': 'United States',
    'Korea (Republic of)': 'Korea, Rep.',
    'Hong Kong, China (SAR)': 'Hong Kong SAR, China',
    'Russian Federation': 'Russian Federation',
    'Viet Nam': 'Vietnam',
    'Congo (Democratic Republic of the)': 'Congo, Dem. Rep.',
    'Congo': 'Congo, Rep.',
    'Egypt': 'Egypt, Arab Rep.',
    'Iran (Islamic Republic of)': 'Iran, Islamic Rep.',
    'Kyrgyzstan': 'Kyrgyz Republic',
    'Lao People\'s Democratic Republic': 'Lao PDR',
    'Moldova (Republic of)': 'Moldova',
    'Palestine, State of': 'West Bank and Gaza',
    'Tanzania (United Republic of)': 'Tanzania',
    'Venezuela (Bolivarian Republic of)': 'Venezuela, RB',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Eswatini (Kingdom of)': 'Eswatini',
    'Syrian Arab Republic': 'Syrian Arab Republic',
    'Korea (Democratic People\'s Rep. of)': "Korea, Dem. People's Rep.",
    'Czechia': 'Czechia',
    'North Macedonia': 'North Macedonia',
    'Cabo Verde': 'Cabo Verde',
}

# First, try direct merge
df = country_res.merge(hdi_data[['Country', 'HDI_Value', 'HDI_Rank', 'HDI_Rank_2022',
                                  'Life_Expectancy', 'Expected_Schooling',
                                  'Mean_Schooling', 'GNI_per_capita', 'GNI_HDI_Rank_Diff']],
                       left_on="Country Name",
                       right_on="Country",
                       how="left")

# If countries are missing, try with mapping
missing_countries = df[df['HDI_Value'].isna()]['Country Name'].tolist()
for panel_country in missing_countries:
    for hdi_country, panel_match in hdi_country_mapping.items():
        if panel_country == panel_match:
            # Get HDI data for this country
            hdi_row = hdi_data[hdi_data['Country'] == hdi_country]
            if not hdi_row.empty:
                # Update the merged data
                mask = df['Country Name'] == panel_country
                for col in hdi_row.columns:
                    if col in df.columns and col != 'Country':
                        df.loc[mask, col] = hdi_row[col].values[0]
            break

# Drop the duplicate Country column
if 'Country' in df.columns:
    df = df.drop(columns=['Country'])

print(f"\nHDI merge results:")
print(f"  Countries with HDI: {df['HDI_Value'].notna().sum()}")
print(f"  Countries missing HDI: {df['HDI_Value'].isna().sum()}")

# Show which countries are missing HDI
missing_hdi = df[df['HDI_Value'].isna()]['Country Name'].tolist()
if missing_hdi:
    print(f"  Missing HDI for: {missing_hdi}")

# ── 5. Merge WGI and GINI ────────────────────────────────────────────────────
wgi  = pd.read_csv(WGI_PATH)
gini = pd.read_csv(GINI_PATH)

df = df.merge(wgi,  on="Country Code", how="left", suffixes=("", "_wgi"))
df = df.merge(gini, on="Country Code", how="left")

print(f"\nAfter WGI/GINI merge:")
print(f"  WGI missing:  {df['WGI_composite'].isna().sum()} countries")
print(f"  GINI missing: {df['GINI_mean'].isna().sum()} countries")

# Drop countries missing key variables
df_clean = df.dropna(subset=["WGI_composite", "GINI_mean"]).copy()
print(f"  Complete cases: {len(df_clean)} countries")

# ── 6. Create HDI categories ──────────────────────────────────────────────
def hdi_category(value):
    if pd.isna(value):
        return 'Missing'
    elif value >= 0.8:
        return 'Very_High'
    elif value >= 0.7:
        return 'High'
    elif value >= 0.55:
        return 'Medium'
    else:
        return 'Low'

df_clean['HDI_Category'] = df_clean['HDI_Value'].apply(hdi_category)

# Create HDI category dummies (drop 'Very_High' as base, and 'Missing' if present)
hdi_cat_dummies = pd.get_dummies(df_clean['HDI_Category'], prefix='HDI_Cat')
# Drop 'Missing' and 'Very_High' as base categories
hdi_cat_dummies = hdi_cat_dummies.drop(columns=['HDI_Cat_Missing'], errors='ignore')
hdi_cat_dummies = hdi_cat_dummies.drop(columns=['HDI_Cat_Very_High'], errors='ignore')

# ── 7. Create final dataset ──────────────────────────────────────────────────
df_final = pd.concat([df_clean, hdi_cat_dummies], axis=1)

# ── 8. Save ──────────────────────────────────────────────────────────────────
cols_to_keep = [
    "Country Name", "Country Code", "n_obs",
    "mean_residual", "mean_che", "mean_oop", "mean_gov",
    "year_min", "year_max",
    "WGI_composite", "WGI_GE", "WGI_CC", "WGI_RL",
    "GINI_mean",
    "HDI_Value", "HDI_Rank", "HDI_Rank_2022", "GNI_per_capita", "GNI_HDI_Rank_Diff",
    "Life_Expectancy", "Expected_Schooling", "Mean_Schooling",
    "HDI_Category"
] + [col for col in hdi_cat_dummies.columns if col in df_final.columns]

# Only keep columns that exist
cols_to_keep = [col for col in cols_to_keep if col in df_final.columns]
df_final[cols_to_keep].to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {OUTPUT_PATH}")

# ── 9. Preview ───────────────────────────────────────────────────────────────
print("\n--- Sample with HDI controls ---")
preview_cols = ["Country Name", "HDI_Value", "HDI_Category",
                "mean_residual", "WGI_composite", "GINI_mean"]
preview_cols = [col for col in preview_cols if col in df_final.columns]
print(df_final[preview_cols].head(20).to_string(index=False))

print("\n--- HDI distribution ---")
print(df_final['HDI_Category'].value_counts())

print("\n--- HDI descriptive statistics ---")
print(df_final['HDI_Value'].describe().round(3))

print("\n--- Countries still missing HDI ---")
missing_hdi = df_final[df_final['HDI_Value'].isna()][['Country Name', 'Country Code']]
if not missing_hdi.empty:
    print(missing_hdi.to_string(index=False))
    print(f"\nTotal missing HDI: {len(missing_hdi)} countries")
else:
    print("All countries have HDI values!")

print("\n--- Descriptive statistics by HDI category ---")
group_cols = ['mean_residual', 'WGI_composite', 'GINI_mean']
group_cols = [col for col in group_cols if col in df_final.columns]
print(df_final.groupby('HDI_Category')[group_cols].mean().round(3))