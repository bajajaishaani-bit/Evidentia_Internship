"""
Efficiency Frontier Visualization for Developing Countries
Lower-Middle and Low-Income Countries Analysis

This script imports functions from the developing countries analysis program
and creates visualizations of the DEA efficiency frontier.

NO REGRESSION IS RUN - only visualizations from existing data
"""

"""
Efficiency Frontier Visualization for Developing Countries
Lower-Middle and Low-Income Countries Analysis

This script imports functions from the developing countries analysis program
and creates visualizations of the DEA efficiency frontier.

NO REGRESSION IS RUN - only visualizations from existing data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Import from developing countries program
# Make sure your file is named: Lower_Mid_Income_countries.py (with underscores)
from Lower_Mid_Income_countries import (
    build_stage2_panel,
    UPPER_MIDDLE_THRESH,
    LOWER_MIDDLE_THRESH,
    LOW_INCOME_THRESH,
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

def plot_frontier_distribution(panel_full, panel_dev, panel_upper_middle, panel_low, save=True):
    """
    Plot distribution of FPE scores with frontier highlighted for developing countries
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full sample distribution (reference)
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

    # 2. Lower-middle + Low income distribution
    ax2 = axes[0, 1]
    frontier_dev = panel_dev[panel_dev['fpe_score'] >= 0.9999]
    non_frontier_dev = panel_dev[panel_dev['fpe_score'] < 0.9999]

    ax2.hist(non_frontier_dev['fpe_score'], bins=30, alpha=0.7,
             color='seagreen', label='Inefficient DMUs', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label=f'Frontier (n={len(frontier_dev)})')
    ax2.set_xlabel('FPE Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Lower-Middle + Low Income: FPE Distribution\n(n={len(panel_dev)})',
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Kernel density comparison
    ax3 = axes[1, 0]
    if len(non_frontier_full) > 0:
        sns.kdeplot(non_frontier_full['fpe_score'], label='Full Sample',
                    ax=ax3, linewidth=2.5)
    if len(non_frontier_dev) > 0:
        sns.kdeplot(non_frontier_dev['fpe_score'], label='Lower-Middle + Low',
                    ax=ax3, linewidth=2.5)
    if len(panel_upper_middle) > 0:
        frontier_upper = panel_upper_middle[panel_upper_middle['fpe_score'] < 0.9999]
        if len(frontier_upper) > 0:
            sns.kdeplot(frontier_upper['fpe_score'], label='Upper-Middle',
                        ax=ax3, linewidth=2.5)
    ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')
    ax3.set_xlabel('FPE Score', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Kernel Density Comparison by Income Group', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = []
    labels = []

    if len(non_frontier_full) > 0:
        data_to_plot.append(non_frontier_full['fpe_score'].values)
        labels.append('Full')
    if len(non_frontier_dev) > 0:
        data_to_plot.append(non_frontier_dev['fpe_score'].values)
        labels.append('Lower-Mid+Low')
    if len(panel_upper_middle) > 0:
        upper_non_frontier = panel_upper_middle[panel_upper_middle['fpe_score'] < 0.9999]
        if len(upper_non_frontier) > 0:
            data_to_plot.append(upper_non_frontier['fpe_score'].values)
            labels.append('Upper-Mid')

    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['steelblue', 'seagreen', 'orange']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')
        ax4.set_ylabel('FPE Score', fontsize=11)
        ax4.set_title('Distribution Comparison (Non-Frontier)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/developing_frontier_distribution.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/developing_frontier_distribution.png")
    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 2: FRONTIER DMUs OVER TIME
# =============================================================================

def plot_frontier_trends(panel_full, panel_dev, panel_upper_middle, panel_low, save=True):
    """
    Plot frontier DMUs over time for developing countries
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

    # 2. Lower-middle + Low income - frontier proportion over time
    ax2 = axes[0, 1]
    yearly_dev = panel_dev.groupby('year').agg({
        'fpe_score': ['mean', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    yearly_dev.columns = ['year', 'mean_fpe', 'n_obs', 'n_frontier']
    yearly_dev['frontier_pct'] = (yearly_dev['n_frontier'] / yearly_dev['n_obs'] * 100)

    ax2.plot(yearly_dev['year'], yearly_dev['frontier_pct'],
             marker='s', linewidth=2.5, color='seagreen', label='Lower-Mid + Low')
    ax2.fill_between(yearly_dev['year'], 0, yearly_dev['frontier_pct'],
                     alpha=0.2, color='seagreen')
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Frontier DMUs (%)', fontsize=11)
    ax2.set_title('Developing Countries: Frontier Proportion Over Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Comparison of frontier proportions across income groups
    ax3 = axes[1, 0]
    ax3.plot(yearly_full['year'], yearly_full['frontier_pct'],
             marker='o', linewidth=2, color='steelblue', label='Full Sample', alpha=0.7)
    ax3.plot(yearly_dev['year'], yearly_dev['frontier_pct'],
             marker='s', linewidth=2, color='seagreen', label='Lower-Mid + Low', alpha=0.7)

    # Add upper-middle if available
    if len(panel_upper_middle) > 0:
        yearly_upper = panel_upper_middle.groupby('year').agg({
            'fpe_score': ['mean', 'count'],
            'is_frontier': lambda x: (x >= 0.9999).sum()
        }).reset_index()
        yearly_upper.columns = ['year', 'mean_fpe', 'n_obs', 'n_frontier']
        yearly_upper['frontier_pct'] = (yearly_upper['n_frontier'] / yearly_upper['n_obs'] * 100)
        ax3.plot(yearly_upper['year'], yearly_upper['frontier_pct'],
                 marker='^', linewidth=2, color='orange', label='Upper-Middle', alpha=0.7)

    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Frontier DMUs (%)', fontsize=11)
    ax3.set_title('Comparison: Frontier Proportion Over Time', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Mean FPE comparison
    ax4 = axes[1, 1]
    ax4.plot(yearly_full['year'], yearly_full['mean_fpe'],
             marker='o', linewidth=2, color='steelblue', label='Full Sample', alpha=0.7)
    ax4.plot(yearly_dev['year'], yearly_dev['mean_fpe'],
             marker='s', linewidth=2, color='seagreen', label='Lower-Mid + Low', alpha=0.7)

    if len(panel_upper_middle) > 0:
        ax4.plot(yearly_upper['year'], yearly_upper['mean_fpe'],
                 marker='^', linewidth=2, color='orange', label='Upper-Middle', alpha=0.7)

    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Frontier')
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Mean FPE Score', fontsize=11)
    ax4.set_title('Mean FPE Scores Over Time', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/developing_frontier_trends.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/developing_frontier_trends.png")
    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 3: FRONTIER VS INEFFICIENT - SCATTER PLOTS
# =============================================================================

def plot_frontier_scatter(panel_full, panel_dev, panel_upper_middle, save=True):
    """
    Plot frontier vs inefficient DMUs with governance indicators for developing countries
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

    # 2. Developing countries: WGI vs FPE
    ax2 = axes[0, 1]
    frontier_dev = panel_dev[panel_dev['fpe_score'] >= 0.9999]
    non_frontier_dev = panel_dev[panel_dev['fpe_score'] < 0.9999]

    ax2.scatter(non_frontier_dev['wgi_composite'], non_frontier_dev['fpe_score'],
                alpha=0.5, s=30, color='seagreen', label='Inefficient DMUs')
    ax2.scatter(frontier_dev['wgi_composite'], frontier_dev['fpe_score'],
                alpha=0.8, s=100, color='red', marker='*', label='Frontier DMUs')
    ax2.set_xlabel('WGI Composite (Governance)', fontsize=11)
    ax2.set_ylabel('FPE Score', fontsize=11)
    ax2.set_title('Developing Countries: Governance vs FPE', fontsize=12)
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

    # 4. Developing countries: GINI vs FPE
    ax4 = axes[1, 1]
    if 'gini' in panel_dev.columns and not panel_dev['gini'].isna().all():
        gini_dev = panel_dev.dropna(subset=['gini'])
        gini_frontier_dev = gini_dev[gini_dev['fpe_score'] >= 0.9999]
        gini_non_frontier_dev = gini_dev[gini_dev['fpe_score'] < 0.9999]

        ax4.scatter(gini_non_frontier_dev['gini'], gini_non_frontier_dev['fpe_score'],
                    alpha=0.5, s=30, color='seagreen', label='Inefficient DMUs')
        ax4.scatter(gini_frontier_dev['gini'], gini_frontier_dev['fpe_score'],
                    alpha=0.8, s=100, color='red', marker='*', label='Frontier DMUs')
        ax4.set_xlabel('GINI Coefficient', fontsize=11)
        ax4.set_ylabel('FPE Score', fontsize=11)
        ax4.set_title('Developing Countries: Inequality vs FPE', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/developing_frontier_scatter.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/developing_frontier_scatter.png")
    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 4: COUNTRY FRONTIER RANKINGS
# =============================================================================

def plot_country_rankings(panel_dev, title="Developing Countries", save=True):
    """
    Plot country rankings by average FPE score for developing countries
    """
    # Calculate average FPE by country
    country_avg = panel_dev.groupby('iso3').agg({
        'fpe_score': ['mean', 'std', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    country_avg.columns = ['iso3', 'mean_fpe', 'std_fpe', 'n_obs', 'n_frontier']
    country_avg = country_avg.sort_values('mean_fpe', ascending=False)
    country_avg['frontier_pct'] = (country_avg['n_frontier'] / country_avg['n_obs'] * 100)

    # Only show countries with at least 3 observations
    country_avg = country_avg[country_avg['n_obs'] >= 3]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Top 15 developing countries
    ax1 = axes[0]
    top15 = country_avg.head(15)
    colors = ['green' if x > 0.8 else 'seagreen' for x in top15['mean_fpe']]
    bars1 = ax1.barh(top15['iso3'], top15['mean_fpe'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Average FPE Score', fontsize=12)
    ax1.set_title(f'Top 15 {title} by FPE Score', fontsize=12)
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

    # Bottom 15 developing countries
    ax2 = axes[1]
    bottom15 = country_avg.tail(15).sort_values('mean_fpe', ascending=True)
    colors = ['coral' if x < 0.5 else 'seagreen' for x in bottom15['mean_fpe']]
    bars2 = ax2.barh(bottom15['iso3'], bottom15['mean_fpe'], color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Average FPE Score', fontsize=12)
    ax2.set_title(f'Bottom 15 {title} by FPE Score', fontsize=12)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')

    # Add error bars
    ax2.errorbar(bottom15['mean_fpe'], range(len(bottom15)),
                 xerr=bottom15['std_fpe'], fmt='none', color='gray', capsize=2, alpha=0.5)

    # Add frontier percentage annotations
    for i, (idx, row) in enumerate(bottom15.iterrows()):
        if row['frontier_pct'] > 0:
            ax2.text(row['mean_fpe'] + 0.01, i, f' {row["frontier_pct"]:.0f}% frontier',
                     va='center', fontsize=8, color='coral')

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/developing_country_rankings.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/developing_country_rankings.png")
    plt.show()

    return country_avg


# =============================================================================
# VISUALIZATION 5: COMPARISON WITH HIGH-INCOME COUNTRIES
# =============================================================================

def plot_income_comparison(panel_full, panel_dev, panel_high_income=None, save=True):
    """
    Compare FPE scores across income groups
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Prepare data for each income group
    groups = []
    labels = []
    colors = []

    # Full sample
    groups.append(panel_full['fpe_score'].values)
    labels.append('All Countries')
    colors.append('steelblue')

    # Developing countries (if available)
    if len(panel_dev) > 0:
        groups.append(panel_dev['fpe_score'].values)
        labels.append('Developing\n(Lower-Mid + Low)')
        colors.append('seagreen')

    # High-income (if provided)
    if panel_high_income is not None and len(panel_high_income) > 0:
        groups.append(panel_high_income['fpe_score'].values)
        labels.append('High-Income')
        colors.append('coral')

    # 1. Box plot comparison
    ax1 = axes[0]
    bp = ax1.boxplot(groups, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')
    ax1.set_ylabel('FPE Score', fontsize=12)
    ax1.set_title('FPE Score Distribution by Income Group', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Violin plot comparison
    ax2 = axes[1]
    data_list = []
    label_list = []
    for i, group in enumerate(groups):
        data_list.append(group)
        label_list.append(labels[i].replace('\n', ' '))

    parts = ax2.violinplot(data_list, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax2.set_xticks(range(1, len(label_list) + 1))
    ax2.set_xticklabels(label_list)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')
    ax2.set_ylabel('FPE Score', fontsize=12)
    ax2.set_title('Violin Plot Comparison', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Summary statistics table
    ax3 = axes[2]
    ax3.axis('tight')
    ax3.axis('off')

    stats_data = []
    for i, (group, label, color) in enumerate(zip(groups, labels, colors)):
        stats_data.append([
            label.replace('\n', ' '),
            f"{len(group):.0f}",
            f"{np.mean(group):.3f}",
            f"{np.std(group):.3f}",
            f"{np.min(group):.3f}",
            f"{np.max(group):.3f}",
            f"{(np.sum(group >= 0.9999) / len(group) * 100):.1f}%"
        ])

    columns = ['Income Group', 'N', 'Mean', 'Std', 'Min', 'Max', '% Frontier']
    table = ax3.table(cellText=stats_data, colLabels=columns,
                      cellLoc='center', loc='center',
                      colWidths=[0.2, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color the header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title('Summary Statistics', fontsize=12, pad=20)

    plt.tight_layout()
    if save:
        plt.savefig('figures/income_group_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/income_group_comparison.png")
    plt.show()

    return fig


# =============================================================================
# MAIN VISUALIZATION PIPELINE
# =============================================================================

def visualize_frontier():
    """
    Main function to run all frontier visualizations for developing countries
    """
    print("\n" + "=" * 65)
    print("  EFFICIENCY FRONTIER VISUALIZATION")
    print("  DEVELOPING COUNTRIES ANALYSIS")
    print("  Lower-Middle and Low Income Countries")
    print("=" * 65)

    # Create figures directory
    import os
    os.makedirs('figures', exist_ok=True)

    # Load data using imported function
    print("\n  Loading data from developing countries program...")
    panel_full, panel_dev, panel_upper_middle, panel_low = build_stage2_panel()

    # Add frontier flags
    panel_full['is_frontier'] = panel_full['fpe_score'] >= 0.9999
    panel_dev['is_frontier'] = panel_dev['fpe_score'] >= 0.9999
    panel_upper_middle['is_frontier'] = panel_upper_middle['fpe_score'] >= 0.9999
    panel_low['is_frontier'] = panel_low['fpe_score'] >= 0.9999

    print(f"\n  Data loaded:")
    print(f"    Full sample: {len(panel_full)} observations, {panel_full['iso3'].nunique()} countries")
    print(f"    Lower-middle + low: {len(panel_dev)} observations, {panel_dev['iso3'].nunique()} countries")
    print(f"    Upper-middle: {len(panel_upper_middle)} observations, {panel_upper_middle['iso3'].nunique()} countries")
    print(f"    Low income: {len(panel_low)} observations, {panel_low['iso3'].nunique()} countries")

    # Generate visualizations
    print("\n  Generating visualizations...")

    # 1. Frontier distribution
    print("\n  1. Frontier Distribution")
    plot_frontier_distribution(panel_full, panel_dev, panel_upper_middle, panel_low, save=True)

    # 2. Frontier trends
    print("\n  2. Frontier Trends Over Time")
    plot_frontier_trends(panel_full, panel_dev, panel_upper_middle, panel_low, save=True)

    # 3. Frontier scatter plots
    print("\n  3. Frontier Scatter Plots")
    plot_frontier_scatter(panel_full, panel_dev, panel_upper_middle, save=True)

    # 4. Country rankings
    print("\n  4. Country Rankings - Developing Countries")
    country_avg = plot_country_rankings(panel_dev, "Developing Countries", save=True)

    # 5. Income group comparison (if high-income data available)
    print("\n  5. Income Group Comparison")
    try:
        # Try to import high-income panel for comparison
        from High_Income_countries_test import build_stage2_panel as build_high_income_panel
        _, _, panel_high_income = build_high_income_panel()
        panel_high_income['is_frontier'] = panel_high_income['fpe_score'] >= 0.9999
        plot_income_comparison(panel_full, panel_dev, panel_high_income, save=True)
    except ImportError:
        print("    High-income data not available - skipping comparison")
        plot_income_comparison(panel_full, panel_dev, None, save=True)

    # Save country rankings
    if country_avg is not None:
        country_avg.to_csv('figures/developing_country_rankings.csv', index=False)
        print(f"  ✓ Country rankings saved to: figures/developing_country_rankings.csv")

    print("\n" + "=" * 65)
    print("  ✅ ALL VISUALIZATIONS COMPLETE")
    print(f"  Visualizations saved to: figures/")
    print("=" * 65)

    return panel_full, panel_dev, country_avg


# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == '__main__':
    # Run the visualization pipeline
    panel_full, panel_dev, country_rankings = visualize_frontier()