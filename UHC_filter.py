"""
Add UHC SCI benchmark eligibility filter to FPE summary.
Run AFTER downloading UHC SCI CSV from WHO data portal.

UHC SCI threshold for benchmark eligibility: >= 50
Rationale: Countries below 50 likely have low CHE10 due to care
avoidance / system absence rather than genuine financial protection
efficiency. Threshold consistent with WHO UHC monitoring literature.
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

# =============================================================================
# LOAD UHC SCI DATA
# Update filename if yours differs
# =============================================================================

def load_uhc_sci(filepath, year_start=2000, year_end=2021):
    """
    Load UHC SCI data. Handles two common WHO export formats:
    Format A: wide — Location | 2023 | 2022 | ... (WHO GHO export)
    Format B: long — Location | Year | Value
    """
    df = pd.read_csv(filepath)
    print(f"UHC SCI file loaded: {df.shape}, columns: {df.columns.tolist()[:8]}")
    print(f"First few rows of UHC data:")
    print(df.head())

    # Detect format
    year_cols = [c for c in df.columns if str(c).isdigit() and
                 year_start <= int(c) <= year_end]

    if year_cols:
        # Wide format
        print(f"Detected wide format with year columns: {year_cols[:5]}...")
        # Detect country code column
        code_col = next((c for c in df.columns
                         if c in ['Location', 'SpatialDimValueCode',
                                  'Country Code', 'iso3', 'CODE']), df.columns[0])
        long = df[[code_col] + year_cols].melt(
            id_vars=code_col, var_name='year', value_name='uhc_sci'
        )
        long = long.rename(columns={code_col: 'location'})
        long['year'] = long['year'].astype(int)
        long['uhc_sci'] = pd.to_numeric(long['uhc_sci'], errors='coerce')

    else:
        # Try long format
        print("Detected long format or trying to parse...")
        loc_col   = next((c for c in df.columns
                          if 'Location' in c or 'Country' in c or 'location' in c.lower()), df.columns[0])
        year_col  = next((c for c in df.columns
                          if 'Period' in c or 'Year' in c or 'year' in c.lower() or 'Time' in c), None)
        val_col   = next((c for c in df.columns
                          if 'Value' in c or 'SCI' in c or 'value' in c.lower()), None)
        if not (year_col and val_col):
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError("Cannot detect year/value columns in UHC SCI file")
        long = df[[loc_col, year_col, val_col]].copy()
        long.columns = ['location', 'year', 'uhc_sci']
        long['year']    = pd.to_numeric(long['year'],    errors='coerce')
        long['uhc_sci'] = pd.to_numeric(long['uhc_sci'], errors='coerce')
        long = long[long['year'].between(year_start, year_end)]

    long = long.dropna(subset=['uhc_sci'])
    print(f"UHC SCI: {long['location'].nunique()} locations, "
          f"{len(long)} obs, years {long['year'].min()}-{long['year'].max()}")
    print(f"Sample locations in UHC data: {long['location'].head(10).tolist()}")
    return long


def build_uhc_eligibility(uhc_long, fpe_iso3_list, threshold=50):
    """
    For each iso3 in the FPE panel, compute:
    - mean UHC SCI across available years
    - benchmark_eligible: True if mean UHC SCI >= threshold

    UHC SCI uses country names not ISO3 codes, so we do a fuzzy match.
    """
    # Build name-to-iso3 map for countries in our FPE panel
    # Standard WHO country name -> ISO3 mapping for common cases
    NAME_TO_ISO3 = {
        'Afghanistan': 'AFG', 'Albania': 'ALB', 'Algeria': 'DZA',
        'Angola': 'AGO', 'Argentina': 'ARG', 'Armenia': 'ARM',
        'Australia': 'AUS', 'Austria': 'AUT', 'Azerbaijan': 'AZE',
        'Bangladesh': 'BGD', 'Belarus': 'BLR', 'Belgium': 'BEL',
        'Benin': 'BEN', 'Bhutan': 'BTN', 'Bolivia': 'BOL',
        'Bolivia (Plurinational State of)': 'BOL',
        'Bosnia and Herzegovina': 'BIH', 'Botswana': 'BWA',
        'Brazil': 'BRA', 'Bulgaria': 'BGR', 'Burkina Faso': 'BFA',
        'Burundi': 'BDI', 'Cambodia': 'KHM', 'Cameroon': 'CMR',
        'Canada': 'CAN', 'Cape Verde': 'CPV', 'Cabo Verde': 'CPV',
        'Chad': 'TCD', 'Chile': 'CHL', 'China': 'CHN',
        'Colombia': 'COL', 'Congo': 'COG',
        'Democratic Republic of the Congo': 'COD',
        "Côte d'Ivoire": 'CIV', 'Croatia': 'HRV', 'Cuba': 'CUB',
        'Czech Republic': 'CZE', 'Czechia': 'CZE',
        'Dominican Republic': 'DOM', 'Ecuador': 'ECU',
        'Egypt': 'EGY', 'El Salvador': 'SLV', 'Estonia': 'EST',
        'Ethiopia': 'ETH', 'Fiji': 'FJI', 'Finland': 'FIN',
        'France': 'FRA', 'Gabon': 'GAB', 'Gambia': 'GMB',
        'Georgia': 'GEO', 'Germany': 'DEU', 'Ghana': 'GHA',
        'Greece': 'GRC', 'Guatemala': 'GTM', 'Guinea': 'GIN',
        'Guinea-Bissau': 'GNB', 'Guyana': 'GUY', 'Haiti': 'HTI',
        'Honduras': 'HND', 'Hungary': 'HUN', 'India': 'IND',
        'Indonesia': 'IDN', 'Iran': 'IRN',
        'Iran (Islamic Republic of)': 'IRN',
        'Iraq': 'IRQ', 'Ireland': 'IRL', 'Israel': 'ISR',
        'Italy': 'ITA', 'Jamaica': 'JAM', 'Japan': 'JPN',
        'Jordan': 'JOR', 'Kazakhstan': 'KAZ', 'Kenya': 'KEN',
        'Kyrgyzstan': 'KGZ', 'Kyrgyz Republic': 'KGZ',
        'Lao PDR': 'LAO', "Lao People's Democratic Republic": 'LAO',
        'Latvia': 'LVA', 'Lebanon': 'LBN', 'Liberia': 'LBR',
        'Libya': 'LBY', 'Lithuania': 'LTU', 'Madagascar': 'MDG',
        'Malawi': 'MWI', 'Malaysia': 'MYS', 'Mali': 'MLI',
        'Mauritania': 'MRT', 'Mauritius': 'MUS', 'Mexico': 'MEX',
        'Moldova': 'MDA', 'Republic of Moldova': 'MDA',
        'Mongolia': 'MNG', 'Montenegro': 'MNE', 'Morocco': 'MAR',
        'Mozambique': 'MOZ', 'Myanmar': 'MMR', 'Namibia': 'NAM',
        'Nepal': 'NPL', 'Netherlands': 'NLD', 'New Zealand': 'NZL',
        'Nicaragua': 'NIC', 'Niger': 'NER', 'Nigeria': 'NGA',
        'North Macedonia': 'MKD', 'Norway': 'NOR', 'Pakistan': 'PAK',
        'Panama': 'PAN', 'Papua New Guinea': 'PNG', 'Paraguay': 'PRY',
        'Peru': 'PER', 'Philippines': 'PHL', 'Poland': 'POL',
        'Portugal': 'PRT', 'Romania': 'ROU', 'Russia': 'RUS',
        'Russian Federation': 'RUS', 'Rwanda': 'RWA',
        'Senegal': 'SEN', 'Serbia': 'SRB', 'Sierra Leone': 'SLE',
        'Singapore': 'SGP', 'Slovakia': 'SVK', 'Slovenia': 'SVN',
        'South Africa': 'ZAF', 'South Sudan': 'SSD', 'Spain': 'ESP',
        'Sri Lanka': 'LKA', 'Sudan': 'SDN', 'Suriname': 'SUR',
        'Sweden': 'SWE', 'Switzerland': 'CHE', 'Syria': 'SYR',
        'Syrian Arab Republic': 'SYR', 'Tajikistan': 'TJK',
        'Tanzania': 'TZA', 'United Republic of Tanzania': 'TZA',
        'Thailand': 'THA', 'Timor-Leste': 'TLS', 'Togo': 'TGO',
        'Tunisia': 'TUN', 'Turkey': 'TUR', 'Turkiye': 'TUR',
        'Uganda': 'UGA', 'Ukraine': 'UKR',
        'United Kingdom': 'GBR',
        'United Kingdom of Great Britain and Northern Ireland': 'GBR',
        'United States': 'USA', 'United States of America': 'USA',
        'Uruguay': 'URY', 'Uzbekistan': 'UZB', 'Venezuela': 'VEN',
        'Venezuela (Bolivarian Republic of)': 'VEN',
        'Viet Nam': 'VNM', 'Vietnam': 'VNM',
        'Yemen': 'YEM', 'Zambia': 'ZMB', 'Zimbabwe': 'ZWE',
    }

    uhc_long = uhc_long.copy()
    uhc_long['iso3'] = uhc_long['location'].map(NAME_TO_ISO3)

    # Show which locations were NOT mapped to ISO3
    unmapped = uhc_long[uhc_long['iso3'].isna()]['location'].unique()
    if len(unmapped) > 0:
        print(f"\nWarning: {len(unmapped)} locations in UHC data couldn't be mapped to ISO3:")
        print(f"  Examples: {unmapped[:5].tolist()}")

    # Mean UHC SCI per country
    mean_uhc = (uhc_long.groupby('iso3')['uhc_sci']
                .mean().reset_index()
                .rename(columns={'uhc_sci': 'mean_uhc_sci'}))

    mean_uhc['benchmark_eligible'] = mean_uhc['mean_uhc_sci'] >= threshold

    # Report
    print(f"\nUHC SCI benchmark eligibility (threshold = {threshold}):")
    matched = mean_uhc[mean_uhc['iso3'].isin(fpe_iso3_list)]
    eligible = matched[matched['benchmark_eligible']]
    ineligible = matched[~matched['benchmark_eligible']]
    print(f"  Eligible for benchmark   : {len(eligible)} countries")
    print(f"  Ineligible (UHC < {threshold}) : {len(ineligible)} countries")
    if len(ineligible) > 0:
        inelig_list = ineligible.sort_values('mean_uhc_sci')
        for _, row in inelig_list.iterrows():
            print(f"    {row['iso3']}: mean UHC SCI = {row['mean_uhc_sci']:.1f}")

    # Show which FPE countries are MISSING from UHC data
    missing = set(fpe_iso3_list) - set(mean_uhc['iso3'].dropna())
    if len(missing) > 0:
        print(f"\nWarning: {len(missing)} countries in FPE data have NO UHC SCI data:")
        print(f"  {sorted(list(missing))[:10]}")

    return mean_uhc[['iso3', 'mean_uhc_sci', 'benchmark_eligible']]


# =============================================================================
# MAIN — run this once UHC SCI CSV is downloaded
# =============================================================================

if __name__ == '__main__':
    import sys

    # CHANGE THIS TO MATCH YOUR ACTUAL FILE NAME
    UHC_FILE  = '/Users/aishaanibajaj/data.csv'   # Changed from UHC_SCI_cleaned.csv to match your warning
    THRESHOLD = 50
    FPE_FILE  = 'FPE_scores.csv'

    # Check if files exist
    if not os.path.exists(FPE_FILE):
        print(f"Error: FPE file '{FPE_FILE}' not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)

    if not os.path.exists(UHC_FILE):
        print(f"\nFile '{UHC_FILE}' not found in current directory.")
        print(f"Current directory: {os.getcwd()}")
        print("Please check:")
        print(f"  - Is the file name exactly '{UHC_FILE}'?")
        print("  - Is it in the same folder as this script?")
        print(f"  - Available files in current directory: {os.listdir('.')}")
        print("\nDownload UHC SCI (SDG 3.8.1) CSV from:")
        print("https://data.who.int/indicators/i/8A1BABA")
        sys.exit(1)

    fpe = pd.read_csv(FPE_FILE)
    fpe_iso3s = fpe['iso3'].unique().tolist()
    print(f"Loaded FPE data with {len(fpe_iso3s)} countries")
    print(f"FPE countries (first 10): {fpe_iso3s[:10]}")

    print(f"\nLoading UHC SCI data from {UHC_FILE}...")
    uhc_long = load_uhc_sci(UHC_FILE)

    eligibility = build_uhc_eligibility(uhc_long, fpe_iso3s, THRESHOLD)

    # Save eligibility lookup for use in fpe_summary.py
    eligibility.to_csv('uhc_benchmark_eligibility.csv', index=False)
    print(f"\nSaved: uhc_benchmark_eligibility.csv")
    print("Now re-run fpe_summary.py — it will automatically use this file")
    print("to exclude care-avoidance countries from benchmark designation.")