"""
Efficiency Frontier Visualization for High-Income Countries
This script imports functions from the high-income analysis program
and creates visualizations of the DEA efficiency frontier.

NO REGRESSION IS RUN - only visualizations from existing data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import from high-income program
from High_Income_countries_test import (
    build_stage2_panel,
    HIGH_INCOME_THRESH,
    YEAR_START,
    YEAR_END
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

# =============================================================================
# VISUALIZATION 1: EFFICIENCY FRONTIER DISTRIBUTION
# =============================================================================

def plot_frontier_distribution(panel_full, panel_high_income, save=True):
    """
    Plot distribution of FPE scores with frontier highlighted
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full sample distribution
    ax1 = axes[0, 0]
    frontier_full = panel_full[panel_full['fpe_score'] >= 0.9999]
    non_frontier_full = panel_full[panel_full['fpe_score'] < 0.9999]

    ax1.hist(non_frontier_full['fpe_score'], bins=30, alpha=0.7,
             color='steelblue', label='Inefficient DMUs', edgecolor='black', linewidth=0.5)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label=f'Frontier (n={len(frontier_full)})')
    ax1.set_xlabel('FPE Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Full Sample: FPE Distribution\n(n={len(panel_full)})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. High-income sample distribution
    ax2 = axes[0, 1]
    frontier_high = panel_high_income[panel_high_income['fpe_score'] >= 0.9999]
    non_frontier_high = panel_high_income[panel_high_income['fpe_score'] < 0.9999]

    ax2.hist(non_frontier_high['fpe_score'], bins=30, alpha=0.7,
             color='seagreen', label='Inefficient DMUs', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label=f'Frontier (n={len(frontier_high)})')
    ax2.set_xlabel('FPE Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'High-Income Sample (GDP ≥ ${HIGH_INCOME_THRESH:,}): FPE Distribution\n(n={len(panel_high_income)})',
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Kernel density comparison
    ax3 = axes[1, 0]
    if len(non_frontier_full) > 0:
        sns.kdeplot(non_frontier_full['fpe_score'], label='Full Sample',
                   ax=ax3, linewidth=2.5)
    if len(non_frontier_high) > 0:
        sns.kdeplot(non_frontier_high['fpe_score'], label='High-Income',
                   ax=ax3, linewidth=2.5)
    ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')
    ax3.set_xlabel('FPE Score', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Kernel Density Comparison', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = []
    labels = []
    if len(non_frontier_full) > 0:
        data_to_plot.append(non_frontier_full['fpe_score'].values)
        labels.append('Full (non-frontier)')
    if len(non_frontier_high) > 0:
        data_to_plot.append(non_frontier_high['fpe_score'].values)
        labels.append('High-Income (non-frontier)')

    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['steelblue', 'seagreen']):
            patch.set_facecolor(color)
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')
        ax4.set_ylabel('FPE Score', fontsize=11)
        ax4.set_title('Distribution Comparison (Non-Frontier)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/frontier_distribution.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/frontier_distribution.png")
    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 2: FRONTIER DMUs OVER TIME
# =============================================================================

def plot_frontier_trends(panel_full, panel_high_income, save=True):
    """
    Plot frontier DMUs over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full sample - frontier proportion over time
    ax1 = axes[0, 0]
    yearly_full = panel_full.groupby('year').agg({
        'fpe_score': ['mean', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    yearly_full.columns = ['year', 'mean_fpe', 'n_obs', 'n_frontier']
    yearly_full['frontier_pct'] = (yearly_full['n_frontier'] / yearly_full['n_obs'] * 100)

    ax1.plot(yearly_full['year'], yearly_full['frontier_pct'],
             marker='o', linewidth=2.5, color='steelblue', label='Full Sample')
    ax1.fill_between(yearly_full['year'], 0, yearly_full['frontier_pct'],
                     alpha=0.2, color='steelblue')
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Frontier DMUs (%)', fontsize=11)
    ax1.set_title('Full Sample: Frontier Proportion Over Time', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. High-income sample - frontier proportion over time
    ax2 = axes[0, 1]
    yearly_high = panel_high_income.groupby('year').agg({
        'fpe_score': ['mean', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    yearly_high.columns = ['year', 'mean_fpe', 'n_obs', 'n_frontier']
    yearly_high['frontier_pct'] = (yearly_high['n_frontier'] / yearly_high['n_obs'] * 100)

    ax2.plot(yearly_high['year'], yearly_high['frontier_pct'],
             marker='s', linewidth=2.5, color='seagreen', label='High-Income')
    ax2.fill_between(yearly_high['year'], 0, yearly_high['frontier_pct'],
                     alpha=0.2, color='seagreen')
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Frontier DMUs (%)', fontsize=11)
    ax2.set_title('High-Income: Frontier Proportion Over Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Comparison of frontier proportions
    ax3 = axes[1, 0]
    ax3.plot(yearly_full['year'], yearly_full['frontier_pct'],
             marker='o', linewidth=2.5, color='steelblue', label='Full Sample', alpha=0.7)
    ax3.plot(yearly_high['year'], yearly_high['frontier_pct'],
             marker='s', linewidth=2.5, color='seagreen', label='High-Income', alpha=0.7)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Frontier DMUs (%)', fontsize=11)
    ax3.set_title('Comparison: Frontier Proportion Over Time', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Mean FPE comparison
    ax4 = axes[1, 1]
    ax4.plot(yearly_full['year'], yearly_full['mean_fpe'],
             marker='o', linewidth=2.5, color='steelblue', label='Full Sample', alpha=0.7)
    ax4.plot(yearly_high['year'], yearly_high['mean_fpe'],
             marker='s', linewidth=2.5, color='seagreen', label='High-Income', alpha=0.7)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Frontier')
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Mean FPE Score', fontsize=11)
    ax4.set_title('Mean FPE Scores Over Time', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/frontier_trends.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/frontier_trends.png")
    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 3: FRONTIER VS INEFFICIENT - SCATTER PLOTS
# =============================================================================

def plot_frontier_scatter(panel_full, panel_high_income, save=True):
    """
    Plot frontier vs inefficient DMUs with governance indicators
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full sample: WGI vs FPE
    ax1 = axes[0, 0]
    frontier_full = panel_full[panel_full['fpe_score'] >= 0.9999]
    non_frontier_full = panel_full[panel_full['fpe_score'] < 0.9999]

    ax1.scatter(non_frontier_full['wgi_composite'], non_frontier_full['fpe_score'],
               alpha=0.5, s=30, color='steelblue', label='Inefficient DMUs')
    ax1.scatter(frontier_full['wgi_composite'], frontier_full['fpe_score'],
               alpha=0.8, s=100, color='red', marker='*', label='Frontier DMUs')
    ax1.set_xlabel('WGI Composite (Governance)', fontsize=11)
    ax1.set_ylabel('FPE Score', fontsize=11)
    ax1.set_title('Full Sample: Governance vs FPE', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. High-income sample: WGI vs FPE
    ax2 = axes[0, 1]
    frontier_high = panel_high_income[panel_high_income['fpe_score'] >= 0.9999]
    non_frontier_high = panel_high_income[panel_high_income['fpe_score'] < 0.9999]

    ax2.scatter(non_frontier_high['wgi_composite'], non_frontier_high['fpe_score'],
               alpha=0.5, s=30, color='seagreen', label='Inefficient DMUs')
    ax2.scatter(frontier_high['wgi_composite'], frontier_high['fpe_score'],
               alpha=0.8, s=100, color='red', marker='*', label='Frontier DMUs')
    ax2.set_xlabel('WGI Composite (Governance)', fontsize=11)
    ax2.set_ylabel('FPE Score', fontsize=11)
    ax2.set_title('High-Income: Governance vs FPE', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Full sample: GINI vs FPE
    ax3 = axes[1, 0]
    if 'gini' in panel_full.columns and not panel_full['gini'].isna().all():
        gini_full = panel_full.dropna(subset=['gini'])
        gini_frontier = gini_full[gini_full['fpe_score'] >= 0.9999]
        gini_non_frontier = gini_full[gini_full['fpe_score'] < 0.9999]

        ax3.scatter(gini_non_frontier['gini'], gini_non_frontier['fpe_score'],
                   alpha=0.5, s=30, color='steelblue', label='Inefficient DMUs')
        ax3.scatter(gini_frontier['gini'], gini_frontier['fpe_score'],
                   alpha=0.8, s=100, color='red', marker='*', label='Frontier DMUs')
        ax3.set_xlabel('GINI Coefficient', fontsize=11)
        ax3.set_ylabel('FPE Score', fontsize=11)
        ax3.set_title('Full Sample: Inequality vs FPE', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. High-income sample: GINI vs FPE
    ax4 = axes[1, 1]
    if 'gini' in panel_high_income.columns and not panel_high_income['gini'].isna().all():
        gini_high = panel_high_income.dropna(subset=['gini'])
        gini_frontier_high = gini_high[gini_high['fpe_score'] >= 0.9999]
        gini_non_frontier_high = gini_high[gini_high['fpe_score'] < 0.9999]

        ax4.scatter(gini_non_frontier_high['gini'], gini_non_frontier_high['fpe_score'],
                   alpha=0.5, s=30, color='seagreen', label='Inefficient DMUs')
        ax4.scatter(gini_frontier_high['gini'], gini_frontier_high['fpe_score'],
                   alpha=0.8, s=100, color='red', marker='*', label='Frontier DMUs')
        ax4.set_xlabel('GINI Coefficient', fontsize=11)
        ax4.set_ylabel('FPE Score', fontsize=11)
        ax4.set_title('High-Income: Inequality vs FPE', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/frontier_scatter.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/frontier_scatter.png")
    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 4: COUNTRY FRONTIER RANKINGS
# =============================================================================

def plot_frontier_rankings(panel_high_income, save=True):
    """
    Plot country rankings by average FPE score for high-income countries
    """
    # Calculate average FPE by country
    country_avg = panel_high_income.groupby('iso3').agg({
        'fpe_score': ['mean', 'std', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    country_avg.columns = ['iso3', 'mean_fpe', 'std_fpe', 'n_obs', 'n_frontier']
    country_avg = country_avg.sort_values('mean_fpe', ascending=False)
    country_avg['frontier_pct'] = (country_avg['n_frontier'] / country_avg['n_obs'] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top 15 high-income countries
    ax1 = axes[0]
    top15 = country_avg.head(15)
    colors = ['green' if x > 0.8 else 'seagreen' for x in top15['mean_fpe']]
    bars1 = ax1.barh(top15['iso3'], top15['mean_fpe'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Average FPE Score', fontsize=12)
    ax1.set_title('Top 15 High-Income Countries by FPE Score', fontsize=12)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')

    # Add error bars
    ax1.errorbar(top15['mean_fpe'], range(len(top15)),
                xerr=top15['std_fpe'], fmt='none', color='gray', capsize=2, alpha=0.5)

    # Add frontier percentage annotations
    for i, (idx, row) in enumerate(top15.iterrows()):
        if row['frontier_pct'] > 0:
            ax1.text(row['mean_fpe'] + 0.01, i, f' {row["frontier_pct"]:.0f}% frontier',
                    va='center', fontsize=8, color='coral')

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom 15 high-income countries
    ax2 = axes[1]
    bottom15 = country_avg.tail(15).sort_values('mean_fpe', ascending=True)
    colors = ['coral' if x < 0.5 else 'seagreen' for x in bottom15['mean_fpe']]
    bars2 = ax2.barh(bottom15['iso3'], bottom15['mean_fpe'], color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Average FPE Score', fontsize=12)
    ax2.set_title('Bottom 15 High-Income Countries by FPE Score', fontsize=12)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')

    # Add error bars
    ax2.errorbar(bottom15['mean_fpe'], range(len(bottom15)),
                xerr=bottom15['std_fpe'], fmt='none', color='gray', capsize=2, alpha=0.5)

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/frontier_country_rankings.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/frontier_country_rankings.png")
    plt.show()

    return country_avg


# =============================================================================
# MAIN VISUALIZATION PIPELINE
# =============================================================================

def visualize_frontier():
    """
    Main function to run all frontier visualizations
    """
    print("\n" + "="*65)
    print("  EFFICIENCY FRONTIER VISUALIZATION")
    print("  HIGH-INCOME COUNTRIES ANALYSIS")
    print("="*65)

    # Create figures directory
    import os
    os.makedirs('figures', exist_ok=True)

    # Load data using imported function
    print("\n  Loading data from high-income program...")
    panel_full, panel_upper_middle, panel_high_income = build_stage2_panel()

    # Add frontier flags
    panel_full['is_frontier'] = panel_full['fpe_score'] >= 0.9999
    panel_high_income['is_frontier'] = panel_high_income['fpe_score'] >= 0.9999

    print(f"\n  Data loaded:")
    print(f"    Full sample: {len(panel_full)} observations, {panel_full['iso3'].nunique()} countries")
    print(f"    Upper-middle + high: {len(panel_upper_middle)} observations, {panel_upper_middle['iso3'].nunique()} countries")
    print(f"    High-income: {len(panel_high_income)} observations, {panel_high_income['iso3'].nunique()} countries")

    # Generate visualizations
    print("\n  Generating visualizations...")

    # 1. Frontier distribution
    print("\n  1. Frontier Distribution")
    plot_frontier_distribution(panel_full, panel_high_income, save=True)

    # 2. Frontier trends
    print("\n  2. Frontier Trends Over Time")
    plot_frontier_trends(panel_full, panel_high_income, save=True)

    # 3. Frontier scatter plots
    print("\n  3. Frontier Scatter Plots")
    plot_frontier_scatter(panel_full, panel_high_income, save=True)

    # 4. Country rankings
    print("\n  4. Country Rankings")
    country_avg = plot_frontier_rankings(panel_high_income, save=True)

    # Save country rankings
    if country_avg is not None:
        country_avg.to_csv('figures/high_income_frontier_rankings.csv', index=False)
        print(f"  ✓ Country rankings saved to: figures/high_income_frontier_rankings.csv")

    print("\n" + "="*65)
    print("  ✅ ALL VISUALIZATIONS COMPLETE")
    print(f"  Visualizations saved to: figures/")
    print("="*65)

    return panel_full, panel_high_income, country_avg


# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == '__main__':
    # Run the visualization pipeline
    panel_full, panel_high_income, country_rankings = visualize_frontier()