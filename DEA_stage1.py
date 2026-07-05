"""
Stage 1: Output-Oriented BCC (VRS) DEA
Paper: Health Financing as Risk Transfer: Measuring Financial Protection
       Efficiency Across Countries

Model:
    Inputs  : x1 = CHE per capita PPP (USD)
              x2 = OOP share (% of current health expenditure)
    Output  : y1 = 100 - CHE10 (financial protection rate)

LP solved for each DMU0 in each year:

    maximize   phi
    subject to:
        sum_j (lambda_j * x1_j) <= x1_0          [CHE per capita constraint]
        sum_j (lambda_j * x2_j) <= x2_0          [OOP share constraint]
        sum_j (lambda_j * y1_j) >= phi * y1_0    [financial protection constraint]
        sum_j (lambda_j)         = 1              [VRS convexity constraint]
        lambda_j >= 0  for all j
        phi      >= 1

    FPE score = 1 / phi*

Variables vector passed to linprog: [lambda_1, ..., lambda_n, phi]
Total variables: n + 1
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_and_melt(filepath, value_name, year_start=2000, year_end=2022):
    """
    Load a wide-format WHO/World Bank CSV and melt to long format.
    Expects: 'Country Name', 'Country Code', year columns as strings.
    Returns: dataframe with columns [iso3, country, year, value_name]
    """
    df = pd.read_csv(filepath)

    # Rename to standard internal names
    df = df.rename(columns={
        'Country Name': 'country',
        'Country Code': 'iso3'
    })

    # Year columns are strings "2000" to year_end
    year_cols = [str(y) for y in range(year_start, year_end + 1)]
    year_cols = [c for c in year_cols if c in df.columns]

    id_cols = ['iso3', 'country']

    df_long = df[id_cols + year_cols].melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name='year',
        value_name=value_name
    )
    df_long['year'] = df_long['year'].astype(int)
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')

    return df_long


def build_panel(che_pc_path, oop_path, che10_path, year_start=2000, year_end=2022):
    """
    Load all three CSVs, merge into a single panel, and construct the output variable.

    Variables:
        che_pc    : CHE per capita PPP (USD)         [input 1]
        oop_share : OOP as % of current health exp   [input 2]
        fp_rate   : 100 - CHE10                      [output]
    """
    df_che_pc  = load_and_melt(che_pc_path,  'che_pc',    year_start, year_end)
    df_oop     = load_and_melt(oop_path,     'oop_share', year_start, year_end)
    df_che10   = load_and_melt(che10_path,   'che10',     year_start, year_end)

    # Merge on iso3, country, year
    merge_cols = ['iso3', 'country', 'year']

    panel = df_che_pc.merge(df_oop,   on=merge_cols, how='outer') \
                     .merge(df_che10, on=merge_cols, how='outer')

    # Construct output: financial protection rate
    panel['fp_rate'] = 100 - panel['che10']

    # Drop rows where any of the three model variables are missing
    panel = panel.dropna(subset=['che_pc', 'oop_share', 'fp_rate'])

    # Basic sanity checks
    panel = panel[panel['fp_rate'] > 0]
    panel = panel[panel['che_pc']  > 0]
    panel = panel[panel['oop_share'] > 0]
    print(df_che_pc['iso3'].nunique())  # how many countries in CHE per capita
    print(df_oop['iso3'].nunique())  # how many in OOP
    print(df_che10['iso3'].nunique())  # how many in CHE10

    panel = panel.reset_index(drop=True)
    print(f"Panel built: {len(panel)} country-year observations across "
          f"{panel['iso3'].nunique()} countries, "
          f"{panel['year'].nunique()} years.")
    return panel


# =============================================================================
# 2. SINGLE-DMU LP SOLVER
# =============================================================================

def solve_dmu(dmu_idx, X, y):
    """
    Solve the output-oriented BCC DEA LP for one DMU.

    Parameters:
        dmu_idx : integer index of DMU0 in the arrays (0-based)
        X       : (n x 2) array of inputs for all n DMUs in this year
        y       : (n,)   array of outputs for all n DMUs in this year

    Returns:
        phi_star  : optimal phi (output expansion factor); None if infeasible
        fpe_score : 1 / phi_star
        lambdas   : (n,) array of optimal peer weights
    """
    n = X.shape[0]  # number of DMUs in this year's cross-section

    # ------------------------------------------------------------------
    # Variables: [lambda_0, lambda_1, ..., lambda_{n-1}, phi]
    # Total: n + 1 variables
    # ------------------------------------------------------------------

    # Objective: minimize -phi  (linprog minimizes, we want to maximize phi)
    # c[0:n] = 0 (lambdas don't appear in objective)
    # c[n]   = -1 (phi coefficient, negated for minimization)
    c = np.zeros(n + 1)
    c[n] = -1.0

    # ------------------------------------------------------------------
    # Inequality constraints: A_ub @ vars <= b_ub
    # We have 3 inequality constraints:
    #   [1] CHE per capita : sum_j lambda_j * x1_j <= x1_0
    #   [2] OOP share      : sum_j lambda_j * x2_j <= x2_0
    #   [3] Output (flipped): -sum_j lambda_j * y_j + phi * y_0 <= 0
    #       (original: sum_j lambda_j * y_j >= phi * y_0
    #        rearranged to <= form: -sum_j lambda_j * y_j + phi*y_0 <= 0)
    # ------------------------------------------------------------------

    x1_0 = X[dmu_idx, 0]  # CHE per capita for DMU0
    x2_0 = X[dmu_idx, 1]  # OOP share for DMU0
    y1_0 = y[dmu_idx]     # financial protection rate for DMU0

    A_ub = np.zeros((3, n + 1))
    b_ub = np.zeros(3)

    # Constraint 1: sum_j lambda_j * x1_j <= x1_0
    A_ub[0, :n] = X[:, 0]   # x1 values for all DMUs
    A_ub[0, n]  = 0.0        # phi doesn't appear here
    b_ub[0]     = x1_0

    # Constraint 2: sum_j lambda_j * x2_j <= x2_0
    A_ub[1, :n] = X[:, 1]   # x2 values for all DMUs
    A_ub[1, n]  = 0.0
    b_ub[1]     = x2_0

    # Constraint 3: -sum_j lambda_j * y_j + phi * y1_0 <= 0
    A_ub[2, :n] = -y          # negated output values
    A_ub[2, n]  = y1_0        # phi coefficient = y1_0
    b_ub[2]     = 0.0

    # ------------------------------------------------------------------
    # Equality constraint: sum_j lambda_j = 1  (VRS convexity)
    # ------------------------------------------------------------------
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0    # lambda coefficients
    A_eq[0, n]  = 0.0    # phi doesn't appear
    b_eq = np.array([1.0])

    # ------------------------------------------------------------------
    # Bounds: lambda_j >= 0, phi >= 1
    # ------------------------------------------------------------------
    bounds = [(0, None)] * n   # lambda_j in [0, inf)
    bounds.append((1, None))   # phi in [1, inf)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    result = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )

    if result.status != 0:
        # LP did not solve to optimality — return None
        return None, None, None

    phi_star  = result.x[n]           # optimal phi
    lambdas   = result.x[:n]          # optimal peer weights
    fpe_score = 1.0 / phi_star        # FPE score in (0, 1]

    return phi_star, fpe_score, lambdas


# =============================================================================
# 3. FULL YEAR DEA
# =============================================================================

def run_year_dea(year_df):
    """
    Run BCC DEA for all DMUs in a single year's cross-section.

    Parameters:
        year_df : dataframe subset for one year, with complete data only

    Returns:
        results_df : dataframe with iso3, country, year,
                     phi, fpe_score, peers (iso3 codes with lambda > 0)
    """
    # Build input matrix X (n x 2) and output vector y (n,)
    X            = year_df[['che_pc', 'oop_share']].values.astype(float)
    y            = year_df['fp_rate'].values.astype(float)
    iso3_list    = year_df['iso3'].values
    country_list = year_df['country'].values
    n            = len(year_df)

    records = []

    for i in range(n):
        phi_star, fpe_score, lambdas = solve_dmu(i, X, y)

        if fpe_score is None:
            # LP failed for this DMU — record as NaN
            records.append({
                'iso3':      iso3_list[i],
                'country':   country_list[i],
                'phi':       np.nan,
                'fpe_score': np.nan,
                'peers':     None
            })
        else:
            # Peer set: countries with lambda > small threshold
            peer_mask = lambdas > 1e-6
            peers     = ','.join(iso3_list[peer_mask])
            records.append({
                'iso3':      iso3_list[i],
                'country':   country_list[i],
                'phi':       phi_star,
                'fpe_score': fpe_score,
                'peers':     peers
            })

    return pd.DataFrame(records)


# =============================================================================
# 4. MASTER LOOP ACROSS ALL YEARS
# =============================================================================

def run_all_years(panel, year_start=2000, year_end=2022):
    """
    Run BCC DEA for every year and collect results.

    Parameters:
        panel      : full panel dataframe from build_panel()
        year_start : first year to process
        year_end   : last year to process

    Returns:
        all_results : dataframe with FPE scores for all country-years
    """
    all_results = []

    for year in range(year_start, year_end + 1):
        year_df = panel[panel['year'] == year].copy().reset_index(drop=True)
        n_dmus  = len(year_df)

        if n_dmus < 3:
            # Too few DMUs to form a meaningful frontier — skip
            print(f"{year}: only {n_dmus} countries with complete data — skipped.")
            continue

        results_df         = run_year_dea(year_df)
        results_df['year'] = year

        # Diagnostics for this year
        n_frontier  = (results_df['fpe_score'].round(6) >= 1.0).sum()
        mean_fpe    = results_df['fpe_score'].mean()
        min_fpe     = results_df['fpe_score'].min()
        n_failed    = results_df['fpe_score'].isna().sum()

        print(f"{year}: {n_dmus} countries | "
              f"{n_frontier} on frontier | "
              f"mean FPE = {mean_fpe:.3f} | "
              f"min FPE = {min_fpe:.3f} | "
              f"{n_failed} LP failures")

        all_results.append(results_df)

    all_results = pd.concat(all_results, ignore_index=True)

    # Reorder columns cleanly
    col_order   = ['iso3', 'country', 'year', 'fpe_score', 'phi', 'peers']
    all_results = all_results[[c for c in col_order if c in all_results.columns]]

    return all_results


# =============================================================================
# 5. MAIN
# =============================================================================

if __name__ == '__main__':

    # ── File paths ──────────────────────────────────────────────────────────
    # Update these to match where your CSV files actually live
    CHE_PC_PATH = '/Users/aishaanibajaj/Downloads/CHE_perCapita_PPP_cleaned.csv'     # CHE per capita PPP (USD)
    OOP_PATH    = '/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /OOP_cleaned.csv'     # OOP share (%)
    CHE10_PATH  = '/Users/aishaanibajaj/Downloads/CHE10_cleaned.csv'   # CHE10 (%)

    YEAR_START = 2000
    YEAR_END   = 2022

    # ── Build panel ─────────────────────────────────────────────────────────
    panel = build_panel(CHE_PC_PATH, OOP_PATH, CHE10_PATH, YEAR_START, YEAR_END)

    # ── Run DEA ─────────────────────────────────────────────────────────────
    print("\nRunning output-oriented BCC DEA by year...\n")
    fpe_scores = run_all_years(panel, YEAR_START, YEAR_END)
    frontier = fpe_scores[fpe_scores['fpe_score'] >= 0.9999]
    print(frontier.groupby('iso3')['year'].count().sort_values(ascending=False).head(15))

    # ── Save output ─────────────────────────────────────────────────────────
    output_path = 'FPE_scores.csv'
    fpe_scores.to_csv(output_path, index=False)
    print(f"\nDone. FPE scores saved to '{output_path}'.")
    print(f"Total country-year observations: {len(fpe_scores)}")
    print(f"\nSample output (first 10 rows):\n")
    print(fpe_scores.head(10).to_string(index=False))
import pandas as pd
fpe_scores = pd.read_csv('FPE_scores.csv')

print("=== Frontier frequency (how often each country appears on frontier) ===")
frontier = fpe_scores[fpe_scores['fpe_score'] >= 0.9999]
print(frontier.groupby(['iso3','country'])['year'].count().sort_values(ascending=False).head(15))

print("\n=== Peer frequency (how often each country is used as a benchmark) ===")
from collections import Counter
peer_counts = Counter()
for peers in fpe_scores['peers'].dropna():
    for p in peers.split(','):
        p = p.strip()
        if p:
            peer_counts[p] += 1
for iso3, count in peer_counts.most_common(15):
    print(f"  {iso3}: {count}")
    # Load World Bank income classification
    # (you can hardcode a small dict or download from WB)
    low_income = ['MLI', 'ZMB', 'ETH', 'RWA', 'BFA', 'MDG', 'MWI', 'UGA', 'MOZ', 'TCD']

    peer_by_income = {
        'low_income': 0,
        'other': 0
    }
    for iso3, count in peer_counts.most_common():
        if iso3 in low_income:
            peer_by_income['low_income'] += count
        else:
            peer_by_income['other'] += count

    print(peer_by_income)