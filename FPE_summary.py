"""
FPE Summary: Income-Group Stratified Efficiency Analysis with UHC SCI Filter
Produces:
  1. Full panel CSV — every country, every year, income group, FPE, UHC SCI, peers
  2. Benchmark table — most consistently eligible frontier country per income group
  3. Efficiency league table within each income group

Benchmark eligibility: country must have UHC SCI >= 50 in that year
(filters out care-avoidance frontier countries like Mali, Ethiopia)
Rwanda eligible from 2013+ when UHC SCI crossed 50.
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

UHC_SCI_THRESHOLD = 50   # WHO SDG 3.8.1 — below this = likely care avoidance

# =============================================================================
# 1. INCOME GROUP CLASSIFICATION (World Bank 2024)
# =============================================================================

INCOME_GROUP = {
    # High Income
    'AUS':'H','AUT':'H','BEL':'H','CAN':'H','CHE':'H','CYP':'H',
    'CZE':'H','DEU':'H','DNK':'H','ESP':'H','EST':'H','FIN':'H',
    'FRA':'H','GBR':'H','GRC':'H','HRV':'H','HUN':'H','IRL':'H',
    'ISL':'H','ISR':'H','ITA':'H','JPN':'H','KOR':'H','LTU':'H',
    'LUX':'H','LVA':'H','MLT':'H','NLD':'H','NOR':'H','NZL':'H',
    'POL':'H','PRT':'H','SVK':'H','SVN':'H','SWE':'H','USA':'H',
    'SGP':'H','ARE':'H','BHR':'H','KWT':'H','QAT':'H','SAU':'H',
    'OMN':'H','HKG':'H','URY':'H','ROU':'H','PAN':'H','TTO':'H',
    'CHL':'H','ARG':'H','AND':'H','BRN':'H',
    # Upper-Middle Income
    'BLR':'UM','BRA':'UM','CHN':'UM','COL':'UM','CRI':'UM','CUB':'UM',
    'DOM':'UM','ECU':'UM','GAB':'UM','GEO':'UM','GTM':'UM','IRN':'UM',
    'IRQ':'UM','JAM':'UM','JOR':'UM','KAZ':'UM','LBN':'UM','MEX':'UM',
    'MKD':'UM','MNE':'UM','MYS':'UM','NAM':'UM','PER':'UM','RUS':'UM',
    'SRB':'UM','THA':'UM','TUN':'UM','TUR':'UM','ZAF':'UM','ALB':'UM',
    'AZE':'UM','BGR':'UM','BIH':'UM','DZA':'UM','MDA':'UM','MNG':'UM',
    'MUS':'UM','PRY':'UM','SLV':'UM','UKR':'UM','ARM':'UM','CPV':'UM',
    'EGY':'UM','MAR':'UM','FJI':'UM','BOL':'UM','IDN':'UM',
    # Lower-Middle Income
    'BGD':'LM','BTN':'LM','CMR':'LM','CIV':'LM','GHA':'LM','HND':'LM',
    'IND':'LM','KEN':'LM','KGZ':'LM','KHM':'LM','LAO':'LM','LKA':'LM',
    'LSO':'LM','MMR':'LM','MRT':'LM','NGA':'LM','NIC':'LM','NPL':'LM',
    'PAK':'LM','PHL':'LM','PNG':'LM','SEN':'LM','TJK':'LM','TLS':'LM',
    'TZA':'LM','UGA':'LM','UZB':'LM','VNM':'LM','ZMB':'LM','ZWE':'LM',
    'DJI':'LM','BEN':'LM','TGO':'LM','GIN':'LM','GMB':'LM','SYR':'LM',
    # Low Income
    'AFG':'L','BDI':'L','BFA':'L','CAF':'L','COD':'L','ERI':'L',
    'ETH':'L','GNB':'L','HTI':'L','LBR':'L','MDG':'L','MLI':'L',
    'MOZ':'L','MWI':'L','NER':'L','RWA':'L','SDN':'L','SLE':'L',
    'SOM':'L','SSD':'L','TCD':'L','YEM':'L',
}

INCOME_LABEL = {
    'H': 'High Income',
    'UM': 'Upper-Middle Income',
    'LM': 'Lower-Middle Income',
    'L':  'Low Income',
}
GROUP_ORDER = ['H', 'UM', 'LM', 'L']

# Countries to explicitly exclude from being benchmarks
EXCLUDED_BENCHMARKS = ['MLI', 'ETH', 'TCD', 'BFA', 'NER', 'COD', 'AFG', 'HTI']


# =============================================================================
# 2. LOAD DATA
# =============================================================================

def load_data(fpe_path='FPE_scores.csv', uhc_path='UHC_SCI_panel.csv'):
    fpe = pd.read_csv(fpe_path)
    fpe['income_group'] = fpe['iso3'].map(INCOME_GROUP)
    fpe['income_label'] = fpe['income_group'].map(INCOME_LABEL).fillna('Unknown')
    fpe['on_frontier']  = (fpe['fpe_score'] >= 0.9999).astype(int)

    # Missing income group warning
    missing = fpe[fpe['income_group'].isna()]['iso3'].unique()
    if len(missing):
        print(f"  Warning: {len(missing)} countries missing income group: {missing}")
        fpe['income_group'] = fpe['income_group'].fillna('Unknown')
        fpe['income_label'] = fpe['income_label'].fillna('Unknown')

    # Load UHC SCI data
    try:
        uhc = pd.read_csv(uhc_path)[['iso3','year','uhc_sci']]
        fpe = fpe.merge(uhc, on=['iso3','year'], how='left')
        fpe['benchmark_eligible'] = fpe['uhc_sci'] >= UHC_SCI_THRESHOLD
        n_eligible = fpe['benchmark_eligible'].sum()
        print(f"  UHC SCI loaded. {n_eligible} country-year obs eligible as benchmark "
              f"(UHC SCI >= {UHC_SCI_THRESHOLD})")
    except FileNotFoundError:
        print(f"  Warning: UHC_SCI_panel.csv not found — applying hard-coded exclusions instead")
        fpe['uhc_sci'] = 100.0
        fpe['benchmark_eligible'] = ~fpe['iso3'].isin(EXCLUDED_BENCHMARKS)
        print(f"  Excluded from benchmark: {EXCLUDED_BENCHMARKS}")

    # Explicitly exclude countries even if UHC data says they're eligible
    fpe.loc[fpe['iso3'].isin(EXCLUDED_BENCHMARKS), 'benchmark_eligible'] = False
    fpe.loc[fpe['iso3'].isin(EXCLUDED_BENCHMARKS), 'uhc_sci'] = 0

    # Remove excluded countries from peer references
    def clean_peers(peers_str):
        if pd.isna(peers_str):
            return peers_str
        peers_list = [p.strip() for p in str(peers_str).split(',')]
        # Remove excluded countries from peer list
        peers_list = [p for p in peers_list if p not in EXCLUDED_BENCHMARKS]
        return ', '.join(peers_list) if peers_list else np.nan

    fpe['peers'] = fpe['peers'].apply(clean_peers)

    return fpe


# =============================================================================
# 3. BENCHMARK COUNTRY PER INCOME GROUP
# =============================================================================

def find_benchmarks(df):
    print(f"\n{'='*70}")
    print("  BENCHMARK COUNTRIES BY INCOME GROUP")
    print(f"  (Eligible = on frontier AND UHC SCI >= {UHC_SCI_THRESHOLD} in that year)")
    print(f"  Explicitly excluded: {EXCLUDED_BENCHMARKS}")
    print(f"{'='*70}")

    results = []

    for grp in GROUP_ORDER:
        grp_df    = df[df['income_group'] == grp]
        if len(grp_df) == 0:
            continue
        label     = INCOME_LABEL[grp]
        n_countries = grp_df['iso3'].nunique()

        # Eligible frontier obs = on frontier AND benchmark_eligible
        elig_frontier = grp_df[
            (grp_df['on_frontier'] == 1) &
            (grp_df['benchmark_eligible'] == True)
        ]

        # Frontier frequency (all, including ineligible) — for reference
        all_frontier_freq = (grp_df[grp_df['on_frontier']==1]
                             .groupby(['iso3','country'])['year']
                             .count().sort_values(ascending=False))

        # Eligible frontier frequency
        elig_freq = (elig_frontier
                     .groupby(['iso3','country'])['year']
                     .count().sort_values(ascending=False))

        # Peer frequency (within group) - EXCLUDING removed countries
        peer_counts = Counter()
        for peers in grp_df['peers'].dropna():
            for p in str(peers).split(','):
                p = p.strip()
                if p and p not in EXCLUDED_BENCHMARKS:
                    peer_counts[p] += 1

        mean_fpe = grp_df['fpe_score'].mean()
        min_fpe  = grp_df['fpe_score'].min()

        print(f"\n  {label} ({n_countries} countries)")
        print(f"  Mean FPE: {mean_fpe:.3f}  |  Min FPE: {min_fpe:.3f}")

        print(f"\n  All frontier countries (incl. ineligible):")
        for (iso3, country), yrs in all_frontier_freq.head(8).items():
            uhc_mean = grp_df[grp_df['iso3']==iso3]['uhc_sci'].mean()
            eligible = grp_df[
                (grp_df['iso3']==iso3) &
                (grp_df['benchmark_eligible']==True) &
                (grp_df['on_frontier']==1)
            ]['year'].count()
            uhc_str = f"UHC≈{uhc_mean:.0f}" if not np.isnan(uhc_mean) else "UHC=n/a"

            excluded_flag = " [EXCLUDED]" if iso3 in EXCLUDED_BENCHMARKS else ""
            flag = "ELIGIBLE" if eligible > 0 else "INELIGIBLE"
            print(f"    {iso3:<6} {country:<28} {yrs:>2} frontier yrs  "
                  f"{uhc_str:<8}  {flag} ({eligible} eligible yrs){excluded_flag}")

        # Best eligible benchmark
        if len(elig_freq) > 0:
            top_iso3    = elig_freq.index[0][0]
            top_country = elig_freq.index[0][1]
            top_yrs     = elig_freq.iloc[0]
            top_peer_n  = peer_counts.get(top_iso3, 0)
            print(f"\n  >>> BENCHMARK: {top_country} ({top_iso3}) — "
                  f"{top_yrs} eligible frontier years, "
                  f"referenced {top_peer_n}x as peer")
        else:
            top_iso3 = top_country = 'None'
            top_yrs  = top_peer_n  = 0
            print(f"\n  >>> No eligible benchmark found for this group "
                  f"(all frontier countries ineligible or below UHC SCI threshold)")

        results.append({
            'income_group':      grp,
            'income_label':      label,
            'benchmark_iso3':    top_iso3,
            'benchmark_country': top_country,
            'eligible_frontier_years': top_yrs,
            'peer_references':   top_peer_n,
            'mean_fpe':          round(mean_fpe, 4),
            'min_fpe':           round(min_fpe, 4),
            'n_countries':       n_countries,
        })

    return pd.DataFrame(results)


# =============================================================================
# 4. COUNTRY EFFICIENCY PANEL
# =============================================================================

def build_country_panel(df):
    records = []

    for iso3, grp_df in df.groupby('iso3'):
        grp_df  = grp_df.sort_values('year')
        country = grp_df['country'].iloc[0]
        inc_grp = grp_df['income_group'].iloc[0]
        inc_lbl = grp_df['income_label'].iloc[0]

        fpe_vals = grp_df['fpe_score'].dropna()
        if len(fpe_vals) == 0:
            continue

        mean_fpe   = fpe_vals.mean()
        min_fpe    = fpe_vals.min()
        max_fpe    = fpe_vals.max()
        n_years    = len(fpe_vals)
        n_frontier = int(grp_df['on_frontier'].sum())
        n_eligible = int(grp_df[
            (grp_df['on_frontier']==1) &
            (grp_df['benchmark_eligible']==True)
        ]['year'].count())
        best_year  = int(grp_df.loc[grp_df['fpe_score'].idxmax(), 'year'])
        worst_year = int(grp_df.loc[grp_df['fpe_score'].idxmin(), 'year'])
        mean_uhc   = grp_df['uhc_sci'].mean()

        # Trend
        if len(fpe_vals) >= 3:
            slope = np.polyfit(grp_df['year'].values, fpe_vals.values, 1)[0]
            trend = 'Improving' if slope > 0.001 else \
                    'Declining' if slope < -0.001 else 'Stable'
        else:
            trend = 'Insufficient data'

        # Top peers (excluding self AND excluded countries)
        all_peers = []
        for peers in grp_df['peers'].dropna():
            all_peers.extend([p.strip() for p in str(peers).split(',')
                               if p.strip() and p.strip() != iso3
                               and p.strip() not in EXCLUDED_BENCHMARKS])
        top_peers = [p for p, _ in Counter(all_peers).most_common(3)]

        records.append({
            'iso3':              iso3,
            'country':           country,
            'income_group':      inc_grp,
            'income_label':      inc_lbl,
            'mean_fpe':          round(mean_fpe, 4),
            'min_fpe':           round(min_fpe, 4),
            'max_fpe':           round(max_fpe, 4),
            'n_years':           n_years,
            'n_frontier_total':  n_frontier,
            'n_frontier_eligible': n_eligible,
            'best_year':         best_year,
            'worst_year':        worst_year,
            'mean_uhc_sci':      round(mean_uhc, 1) if not np.isnan(mean_uhc) else None,
            'trend':             trend,
            'learn_from':        ', '.join(top_peers) if top_peers else '—',
        })

    panel = pd.DataFrame(records)

    # Rank within income group by mean FPE
    panel['rank_in_group'] = (panel
        .groupby('income_group')['mean_fpe']
        .rank(ascending=False, method='min').astype(int))

    grp_sort = {'H':0,'UM':1,'LM':2,'L':3,'Unknown':4}
    panel['_sort'] = panel['income_group'].map(grp_sort).fillna(5)
    panel = panel.sort_values(['_sort','rank_in_group']).drop(columns='_sort')
    return panel


# =============================================================================
# 5. PRINT LEAGUE TABLES
# =============================================================================

def print_league_tables(panel):
    print(f"\n{'='*90}")
    print("  EFFICIENCY LEAGUE TABLE BY INCOME GROUP")
    print(f"{'='*90}")

    for grp in GROUP_ORDER:
        grp_panel = panel[panel['income_group'] == grp]
        if len(grp_panel) == 0:
            continue
        label = INCOME_LABEL[grp]
        print(f"\n  {label} — {len(grp_panel)} countries")
        print(f"  {'Rank':<5} {'ISO3':<6} {'Country':<28} "
              f"{'Mean FPE':>9} {'Frontier':>9} {'UHC SCI':>8} "
              f"{'Trend':<11} {'Learn From'}")
        print(f"  {'-'*95}")

        for _, row in grp_panel.iterrows():
            frontier_str = f"{row['n_frontier_total']}/{row['n_years']}yrs"
            uhc_str      = f"{row['mean_uhc_sci']:.0f}" \
                           if row['mean_uhc_sci'] else '—'
            peers_str    = row['learn_from'][:22] if row['learn_from'] != '—' else '—'

            excluded_mark = " [EXCLUDED]" if row['iso3'] in EXCLUDED_BENCHMARKS else ""

            print(f"  {int(row['rank_in_group']):<5} "
                  f"{row['iso3']:<6} "
                  f"{row['country']:<28} "
                  f"{row['mean_fpe']:>9.4f} "
                  f"{frontier_str:>9} "
                  f"{uhc_str:>8} "
                  f"{row['trend']:<11} "
                  f"{peers_str}{excluded_mark}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("Loading FPE scores and UHC SCI data...")
    df = load_data('FPE_scores.csv', 'UHC_SCI_panel.csv')

    print(f"\n  Total obs    : {len(df)}")
    print(f"  Countries    : {df['iso3'].nunique()}")

    benchmarks = find_benchmarks(df)
    panel      = build_country_panel(df)
    print_league_tables(panel)

    panel.to_csv('FPE_country_panel.csv', index=False)
    benchmarks.to_csv('FPE_benchmarks.csv', index=False)

    print(f"\n  Saved: FPE_country_panel.csv ({len(panel)} countries)")
    print(f"  Saved: FPE_benchmarks.csv ({len(benchmarks)} income groups)")
    print(f"\n  Countries excluded from all outputs: {EXCLUDED_BENCHMARKS}")