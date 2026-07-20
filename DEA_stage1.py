"""
Stage 1: Output-Oriented BCC (VRS) DEA — Income-Stratified with UHC SCI Eligibility Filter
Paper: Health Financing as Risk Transfer: Measuring Financial Protection
       Efficiency Across Countries

Model (unchanged from base spec):
    Inputs  : x1 = CHE per capita PPP (USD)
              x2 = OOP share (% of current health expenditure)
    Output  : y1 = 100 - CHE10 (financial protection rate)

WHAT'S NEW vs. the pooled version:
    1. UHC SCI eligibility filter — countries with UHC Service Coverage
       Index < 50 in a given year are EXCLUDED from the frontier-building
       set for that year. This prevents "care-avoidance" countries (e.g.
       Mali) from anchoring the frontier by looking efficient simply
       because people never sought care in the first place.
       IMPORTANT: filter is applied BEFORE the LP is solved, not after
       scoring — a low-UHC-SCI country cannot appear as a DMU, and
       cannot appear as a peer in anyone else's lambda vector, for that
       year.
    2. Income-group stratification — the frontier is built SEPARATELY
       within each World Bank income group (L / LM / UM / H) per year,
       using year-varying classifications (a country's income group can
       change across the panel). A country is only ever benchmarked
       against other countries in the same income bracket in the same
       year.

LP solved for each DMU0, within its income-group x year cross-section:

    maximize   phi
    subject to:
        sum_j (lambda_j * x1_j) <= x1_0
        sum_j (lambda_j * x2_j) <= x2_0
        sum_j (lambda_j * y1_j) >= phi * y1_0
        sum_j (lambda_j)         = 1              [VRS convexity]
        lambda_j >= 0  for all j
        phi      >= 1

    FPE score = 1 / phi*
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# =============================================================================
# 1. DATA LOADING — health financing variables
# =============================================================================

def load_and_melt(filepath, value_name, year_start=2000, year_end=2022):
    """
    Load a wide-format WHO/World Bank CSV and melt to long format.
    Expects: 'Country Name', 'Country Code', year columns as strings.
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={'Country Name': 'country', 'Country Code': 'iso3'})

    year_cols = [str(y) for y in range(year_start, year_end + 1)]
    year_cols = [c for c in year_cols if c in df.columns]
    id_cols = ['iso3', 'country']

    df_long = df[id_cols + year_cols].melt(
        id_vars=id_cols, value_vars=year_cols,
        var_name='year', value_name=value_name
    )
    df_long['year'] = df_long['year'].astype(int)
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    return df_long


def load_uhc_sci(filepath, year_start=2000, year_end=2022):
    """
    Load UHC Service Coverage Index from a WHO GHO long-format export
    (same style as your CHE10 data.csv — columns include IndicatorCode,
    SpatialDimValueCode, Period, FactValueNumeric, etc.).

    Filters to IndicatorCode == 'UHC_INDEX_REPORTED'.
    """
    df = pd.read_csv(filepath)
    df = df[df['IndicatorCode'] == 'UHC_INDEX_REPORTED'].copy()

    df['iso3'] = df['SpatialDimValueCode']
    df['year'] = pd.to_numeric(df['Period'], errors='coerce')
    df['uhc_sci'] = pd.to_numeric(df['FactValueNumeric'], errors='coerce')

    df = df.dropna(subset=['iso3', 'year', 'uhc_sci'])
    df['year'] = df['year'].astype(int)
    df = df[(df['year'] >= year_start) & (df['year'] <= year_end)]

    # One value per country-year expected; if duplicates exist (e.g. from
    # disaggregation dimensions), take the max as a safety net.
    df = df.groupby(['iso3', 'year'], as_index=False)['uhc_sci'].max()

    return df[['iso3', 'year', 'uhc_sci']]


def load_income_groups(filepath, year_start=2000, year_end=2022):
    """
    Load World Bank historical income classification from the OGHIST.xlsx
    'Country Analytical History' sheet.

    Structure (verified against the actual file):
        - Row index 5 (0-based)  : calendar years, aligned to columns
        - Row index 6+           : iso3 (col 0), country name (col 1),
                                    then one income-group code per
                                    fiscal-year column ('L','LM','UM','H',
                                    or '..' for unclassified)
    """
    raw = pd.read_excel(filepath, sheet_name='Country Analytical History', header=None)

    calendar_year_row = raw.iloc[5]
    data = raw.iloc[6:].copy()
    data = data.rename(columns={0: 'iso3', 1: 'country'})
    data = data.dropna(subset=['iso3'])

    year_col_map = {}
    for col in data.columns[2:]:
        yr = calendar_year_row[col]
        if pd.notna(yr):
            try:
                yr_int = int(yr)
                if year_start <= yr_int <= year_end:
                    year_col_map[col] = yr_int
            except (ValueError, TypeError):
                continue

    if not year_col_map:
        raise ValueError(
            f"No calendar-year columns found in range {year_start}-{year_end}. "
            f"Check that the 'Country Analytical History' sheet structure "
            f"still matches (calendar years expected in row index 5)."
        )

    keep_cols = ['iso3'] + list(year_col_map.keys())
    long_df = data[keep_cols].melt(
        id_vars='iso3', var_name='col', value_name='income_group_raw'
    )
    long_df['year'] = long_df['col'].map(year_col_map)
    long_df = long_df.drop(columns='col')

    long_df['income_group'] = long_df['income_group_raw'].astype(str).str.strip()
    long_df = long_df[long_df['income_group'].isin(['L', 'LM', 'UM', 'H'])]

    return long_df[['iso3', 'year', 'income_group']]


# =============================================================================
# 2. BUILD PANEL — merge financing vars + UHC SCI + income group
# =============================================================================

def build_panel(che_pc_path, oop_path, che10_path, uhc_sci_path,
                 income_group_path, year_start=2000, year_end=2022,
                 uhc_sci_threshold=50):
    """
    Build the full eligibility-filtered, income-tagged panel.
    """
    df_che_pc = load_and_melt(che_pc_path, 'che_pc', year_start, year_end)
    df_oop = load_and_melt(oop_path, 'oop_share', year_start, year_end)
    df_che10 = load_and_melt(che10_path, 'che10', year_start, year_end)
    df_uhc = load_uhc_sci(uhc_sci_path, year_start, year_end)
    df_income = load_income_groups(income_group_path, year_start, year_end)

    merge_cols = ['iso3', 'country', 'year']
    panel = (df_che_pc
             .merge(df_oop, on=merge_cols, how='outer')
             .merge(df_che10, on=merge_cols, how='outer'))

    panel['fp_rate'] = 100 - panel['che10']
    panel = panel.dropna(subset=['che_pc', 'oop_share', 'fp_rate'])
    panel = panel[(panel['fp_rate'] > 0) & (panel['che_pc'] > 0) & (panel['oop_share'] > 0)]

    # Merge UHC SCI — needed for eligibility, iso3+year only (no country col)
    panel = panel.merge(df_uhc[['iso3', 'year', 'uhc_sci']], on=['iso3', 'year'], how='left')

    before_elig = len(panel)
    n_missing_sci = panel['uhc_sci'].isna().sum()

    # ELIGIBILITY FILTER: drop countries below UHC SCI threshold in that year.
    # Countries with missing UHC SCI are also dropped (can't verify eligibility
    # -> excluded rather than assumed eligible).
    panel = panel[panel['uhc_sci'] >= uhc_sci_threshold]
    print(f"Eligibility filter: {before_elig} -> {len(panel)} obs "
          f"(dropped {before_elig - len(panel)}; of which {n_missing_sci} had no UHC SCI value)")

    # Merge income group — iso3+year
    panel = panel.merge(df_income, on=['iso3', 'year'], how='left')
    before_income = len(panel)
    panel = panel.dropna(subset=['income_group'])
    print(f"Income group merge: {before_income} -> {len(panel)} obs "
          f"(dropped {before_income - len(panel)} with no income classification)")

    panel = panel.reset_index(drop=True)
    print(f"\nFinal panel: {len(panel)} country-year observations across "
          f"{panel['iso3'].nunique()} countries, {panel['year'].nunique()} years.")
    print(panel.groupby('income_group')['iso3'].nunique())

    return panel


# =============================================================================
# 3. SINGLE-DMU LP SOLVER (unchanged math, just called within strata now)
# =============================================================================

def solve_dmu(dmu_idx, X, y):
    n = X.shape[0]
    c = np.zeros(n + 1)
    c[n] = -1.0

    x1_0, x2_0, y1_0 = X[dmu_idx, 0], X[dmu_idx, 1], y[dmu_idx]

    A_ub = np.zeros((3, n + 1))
    b_ub = np.zeros(3)
    A_ub[0, :n] = X[:, 0]; b_ub[0] = x1_0
    A_ub[1, :n] = X[:, 1]; b_ub[1] = x2_0
    A_ub[2, :n] = -y; A_ub[2, n] = y1_0; b_ub[2] = 0.0

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * n + [(1, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

    if result.status != 0:
        return None, None, None

    phi_star = result.x[n]
    lambdas = result.x[:n]
    fpe_score = 1.0 / phi_star
    return phi_star, fpe_score, lambdas


# =============================================================================
# 4. RUN DEA WITHIN ONE (YEAR, INCOME GROUP) STRATUM
# =============================================================================

def run_stratum_dea(stratum_df):
    X = stratum_df[['che_pc', 'oop_share']].values.astype(float)
    y = stratum_df['fp_rate'].values.astype(float)
    iso3_list = stratum_df['iso3'].values
    country_list = stratum_df['country'].values
    n = len(stratum_df)

    records = []
    for i in range(n):
        phi_star, fpe_score, lambdas = solve_dmu(i, X, y)
        if fpe_score is None:
            records.append({'iso3': iso3_list[i], 'country': country_list[i],
                             'phi': np.nan, 'fpe_score': np.nan, 'peers': None})
        else:
            peer_mask = lambdas > 1e-6
            peers = ','.join(iso3_list[peer_mask])
            records.append({'iso3': iso3_list[i], 'country': country_list[i],
                             'phi': phi_star, 'fpe_score': fpe_score, 'peers': peers})
    return pd.DataFrame(records)


# =============================================================================
# 5. MASTER LOOP — ACROSS YEARS x INCOME GROUPS
# =============================================================================

def run_all_strata(panel, year_start=2000, year_end=2022, min_dmus=3):
    all_results = []

    for year in range(year_start, year_end + 1):
        year_df = panel[panel['year'] == year]
        if year_df.empty:
            continue

        for income_group in ['L', 'LM', 'UM', 'H']:
            stratum_df = year_df[year_df['income_group'] == income_group] \
                .copy().reset_index(drop=True)
            n_dmus = len(stratum_df)

            if n_dmus < min_dmus:
                print(f"{year} [{income_group}]: only {n_dmus} countries — skipped.")
                continue

            results_df = run_stratum_dea(stratum_df)
            results_df['year'] = year
            results_df['income_group'] = income_group

            n_frontier = (results_df['fpe_score'].round(6) >= 1.0).sum()
            mean_fpe = results_df['fpe_score'].mean()
            n_failed = results_df['fpe_score'].isna().sum()

            print(f"{year} [{income_group}]: {n_dmus} countries | "
                  f"{n_frontier} on frontier | mean FPE = {mean_fpe:.3f} | "
                  f"{n_failed} LP failures")

            all_results.append(results_df)

    all_results = pd.concat(all_results, ignore_index=True)
    col_order = ['iso3', 'country', 'year', 'income_group', 'fpe_score', 'phi', 'peers']
    all_results = all_results[[c for c in col_order if c in all_results.columns]]
    return all_results


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == '__main__':

    CHE_PC_PATH = '/Users/aishaanibajaj/Downloads/CHE_perCapita_PPP_cleaned.csv'
    OOP_PATH = '/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /OOP_cleaned.csv'
    CHE10_PATH = '/Users/aishaanibajaj/Downloads/CHE10_cleaned.csv'
    UHC_SCI_PATH = '/Users/aishaanibajaj/Downloads/data-2.csv'          # <-- update to your actual file
    INCOME_GROUP_PATH = '/Users/aishaanibajaj/Downloads/OGHIST_2026_07_01.xlsx'             # <-- update once downloaded

    YEAR_START, YEAR_END = 2000, 2022
    UHC_SCI_THRESHOLD = 50

    panel = build_panel(CHE_PC_PATH, OOP_PATH, CHE10_PATH, UHC_SCI_PATH,
                         INCOME_GROUP_PATH, YEAR_START, YEAR_END, UHC_SCI_THRESHOLD)

    print("\nRunning income-stratified output-oriented BCC DEA by year...\n")
    fpe_scores = run_all_strata(panel, YEAR_START, YEAR_END)

    print("\n=== Frontier frequency by income group ===")
    frontier = fpe_scores[fpe_scores['fpe_score'] >= 0.9999]
    print(frontier.groupby(['income_group', 'iso3'])['year'].count()
          .sort_values(ascending=False).groupby('income_group').head(5))

    output_path = 'FPE_scores_stratified.csv'
    fpe_scores.to_csv(output_path, index=False)
    print(f"\nDone. FPE scores saved to '{output_path}'.")
    print(f"Total country-year observations: {len(fpe_scores)}")