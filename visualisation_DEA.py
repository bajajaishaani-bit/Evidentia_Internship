"""
DEA Visualizations for FPE Scores
Three plots:
    1. BCC Efficiency Frontier (2D) for a representative year
    2. FPE Score Distribution across years (box plot)
    3. Country Trajectory — FPE evolution over time for selected countries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
# In visualisation_DEA.py
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import from the other file
from DEA_stage1 import build_panel, run_all_years  # or whatever your file is named
# =============================================================================
# LOAD DATA
# =============================================================================

# FPE scores output from Stage 1
fpe = pd.read_csv('FPE_scores.csv')

# Original panel (rebuilt here — adjust paths to match yours)
# We need the raw variables (che_pc, oop_share, fp_rate) for the frontier plot
from DEA_stage1 import build_panel

panel = build_panel(
    che_pc_path='/Users/aishaanibajaj/Downloads/CHE_perCapita_PPP_cleaned.csv',
    oop_path='/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /OOP_cleaned.csv',
    che10_path='/Users/aishaanibajaj/Downloads/Internship Evidentia /First regression /CHE10_cleaned.csv',
    year_start=2000,
    year_end=2022
)

# Merge FPE scores into panel
panel = panel.merge(fpe[['iso3', 'year', 'fpe_score']], on=['iso3', 'year'], how='left')

# Style settings — clean academic look
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150
})

FRONTIER_COLOR = '#2C7BB6'  # blue  — frontier countries
INEFFICIENT_COLOR = '#D7191C'  # red   — inefficient countries
FRONTIER_LINE = '#1A1A2E'  # dark  — frontier line


# =============================================================================
# PLOT 1: BCC EFFICIENCY FRONTIER (2D)
# Choose a year with the most countries for clearest picture
# Fix CHE per capita — plot OOP share (x) vs financial protection rate (y)
# =============================================================================

def plot_frontier(panel, year=None, save_path='plot1_frontier.png'):
    """
    2D frontier plot: OOP share (x-axis) vs Financial Protection Rate (y-axis).
    The BCC frontier is approximated by the piecewise-linear upper-left envelope
    of the efficient DMUs.

    We pick the year with the most observations if year is not specified.
    """
    if year is None:
        year = panel.groupby('year')['iso3'].count().idxmax()

    df = panel[panel['year'] == year].dropna(subset=['fpe_score']).copy()
    df = df.sort_values('oop_share').reset_index(drop=True)

    on_frontier = df[df['fpe_score'] >= 0.9999].copy()
    inefficient = df[df['fpe_score'] < 0.9999].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot inefficient countries
    ax.scatter(
        inefficient['oop_share'], inefficient['fp_rate'],
        color=INEFFICIENT_COLOR, alpha=0.7, s=60, zorder=3,
        label='Inefficient (FPE < 1)'
    )

    # Plot frontier countries
    ax.scatter(
        on_frontier['oop_share'], on_frontier['fp_rate'],
        color=FRONTIER_COLOR, s=80, zorder=4, marker='D',
        label='Efficient (FPE = 1)'
    )

    # Draw piecewise-linear BCC frontier
    # Sort efficient DMUs by OOP share ascending
    # The frontier is the upper-left staircase envelope
    if len(on_frontier) >= 2:
        frontier_sorted = on_frontier.sort_values('oop_share')
        fx = frontier_sorted['oop_share'].values
        fy = frontier_sorted['fp_rate'].values

        # Extend frontier lines to the edges of the plot
        x_min = df['oop_share'].min() * 0.9
        x_max = df['oop_share'].max() * 1.05

        # Draw vertical extensions at both ends
        ax.plot([x_min, fx[0]], [fy[0], fy[0]],
                color=FRONTIER_LINE, lw=1.5, ls='--', alpha=0.5)
        ax.plot([fx[-1], x_max], [fy[-1], fy[-1]],
                color=FRONTIER_LINE, lw=1.5, ls='--', alpha=0.5)

        # Draw the frontier segments connecting efficient DMUs
        ax.plot(fx, fy, color=FRONTIER_LINE, lw=2.0, zorder=2,
                label='BCC Frontier')

    # Label frontier countries by ISO3 code
    for _, row in on_frontier.iterrows():
        ax.annotate(
            row['iso3'],
            (row['oop_share'], row['fp_rate']),
            textcoords='offset points', xytext=(5, 4),
            fontsize=8, color=FRONTIER_COLOR, fontweight='bold'
        )

    # Label a few interesting inefficient countries (lowest FPE)
    worst = inefficient.nsmallest(5, 'fpe_score')
    for _, row in worst.iterrows():
        ax.annotate(
            f"{row['iso3']} ({row['fpe_score']:.2f})",
            (row['oop_share'], row['fp_rate']),
            textcoords='offset points', xytext=(5, -10),
            fontsize=7.5, color=INEFFICIENT_COLOR, alpha=0.85
        )

    ax.set_xlabel('OOP Share (% of Current Health Expenditure)', fontsize=12)
    ax.set_ylabel('Financial Protection Rate (100 − CHE10, %)', fontsize=12)
    ax.set_title(
        f'BCC Efficiency Frontier — Financial Protection, {year}\n'
        f'Output-Oriented VRS DEA  |  n = {len(df)} countries',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.legend(framealpha=0.9, fontsize=10)

    # Annotation explaining the axes
    ax.text(
        0.02, 0.04,
        'Countries above-left of the frontier are efficient.\n'
        'Distance below frontier = unexploited financial protection potential.',
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='grey', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}  (year = {year}, n = {len(df)})")
    plt.show()


# =============================================================================
# PLOT 2: FPE SCORE DISTRIBUTION ACROSS YEARS (Box Plot)
# =============================================================================

def plot_distribution(fpe, save_path='plot2_distribution.png'):
    """
    Box plot showing distribution of FPE scores by year.
    Reveals whether financial protection efficiency is improving,
    declining, or stable across the panel.
    """
    # Build list of FPE arrays per year — only years with >= 5 observations
    years = sorted(fpe['year'].unique())
    data_by_yr = [fpe[fpe['year'] == y]['fpe_score'].dropna().values for y in years]
    labels = [str(y) for y in years]

    # Filter to years with enough data
    min_obs = 5
    filtered = [(y, d) for y, d in zip(labels, data_by_yr) if len(d) >= min_obs]
    labels_f, data_f = zip(*filtered)

    fig, ax = plt.subplots(figsize=(14, 6))

    bp = ax.boxplot(
        data_f,
        patch_artist=True,
        notch=False,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='grey'),
        capprops=dict(color='grey'),
        flierprops=dict(marker='o', color=INEFFICIENT_COLOR,
                        alpha=0.5, markersize=4)
    )

    # Color boxes by mean FPE — darker blue = higher mean efficiency
    means = [np.mean(d) for d in data_f]
    norm = plt.Normalize(min(means) - 0.01, max(means) + 0.01)
    cmap = plt.cm.Blues

    for patch, mean_val in zip(bp['boxes'], means):
        patch.set_facecolor(cmap(norm(mean_val)))
        patch.set_alpha(0.8)

    # Add mean FPE as text above each box
    for i, mean_val in enumerate(means):
        ax.text(i + 1, mean_val + 0.002, f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=7, color='navy')

    # Horizontal reference line at FPE = 1
    ax.axhline(1.0, color='green', lw=1.2, ls='--', alpha=0.7, label='Frontier (FPE = 1)')

    ax.set_xticklabels(labels_f, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('FPE Score', fontsize=12)
    ax.set_title(
        'Distribution of Financial Protection Efficiency (FPE) Scores by Year\n'
        'Output-Oriented BCC DEA  |  Shading = mean FPE level',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.legend(fontsize=10)

    # Add colorbar for mean FPE
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label('Mean FPE', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# PLOT 3: COUNTRY TRAJECTORY — FPE OVER TIME
# Select countries with the most observations and most interesting variation
# =============================================================================

def plot_trajectories(fpe, n_countries=10, save_path='plot3_trajectories.png'):
    """
    Line plot of FPE score over time for selected countries.
    Selects: countries with most data points AND highest FPE variance
    (most interesting trajectories), plus always includes frontier anchors.
    """
    # Countries with enough observations
    obs_count = fpe.groupby('iso3')['fpe_score'].count()
    enough_obs = obs_count[obs_count >= 5].index

    # Among those, pick by highest variance (most dynamic trajectories)
    variance = fpe[fpe['iso3'].isin(enough_obs)].groupby('iso3')['fpe_score'].std()
    top_varied = variance.nlargest(n_countries).index.tolist()

    # Always include countries that frequently appear on frontier
    frontier_counts = (
        fpe[fpe['fpe_score'] >= 0.9999]
        .groupby('iso3')['year'].count()
        .nlargest(3).index.tolist()
    )

    # Combine and deduplicate, cap at n_countries + 3
    selected = list(dict.fromkeys(frontier_counts + top_varied))[:n_countries + 3]

    df_sel = fpe[fpe['iso3'].isin(selected)].copy()

    # Color map — one color per country
    cmap = plt.cm.get_cmap('tab20', len(selected))
    colors = {iso3: cmap(i) for i, iso3 in enumerate(selected)}

    fig, ax = plt.subplots(figsize=(13, 7))

    for iso3, grp in df_sel.groupby('iso3'):
        grp_sorted = grp.sort_values('year')
        country = grp_sorted['country'].iloc[0]
        ax.plot(
            grp_sorted['year'], grp_sorted['fpe_score'],
            marker='o', markersize=4, linewidth=1.8,
            color=colors[iso3], label=f'{iso3} — {country}',
            alpha=0.85
        )

    # Frontier reference
    ax.axhline(1.0, color='black', lw=1.2, ls='--', alpha=0.5, label='Frontier (FPE = 1)')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('FPE Score', fontsize=12)
    ax.set_title(
        'Financial Protection Efficiency Trajectories — Selected Countries\n'
        'Countries selected by data availability and FPE variation',
        fontsize=13, fontweight='bold', pad=12
    )

    ax.set_ylim(
        max(0, df_sel['fpe_score'].min() - 0.05),
        1.03
    )
    ax.set_xticks(sorted(df_sel['year'].unique()))
    ax.tick_params(axis='x', rotation=45)

    # Legend outside plot
    ax.legend(
        loc='upper left', bbox_to_anchor=(1.01, 1),
        fontsize=8.5, framealpha=0.9, borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# RUN ALL THREE PLOTS
# =============================================================================

if __name__ == '__main__':
    # Pick the year with the most DMUs for frontier plot
    best_year = panel.groupby('year')['iso3'].count().idxmax()
    print(f"\nUsing year {best_year} for frontier plot "
          f"({panel[panel['year'] == best_year]['iso3'].count()} countries)\n")

    print("Generating Plot 1: BCC Efficiency Frontier...")
    plot_frontier(panel, year=best_year, save_path='plot1_frontier.png')

    print("\nGenerating Plot 2: FPE Distribution Across Years...")
    plot_distribution(fpe, save_path='plot2_distribution.png')

    print("\nGenerating Plot 3: Country Trajectories...")
    plot_trajectories(fpe, n_countries=10, save_path='plot3_trajectories.png')

    print("\nAll plots saved.")