"""
DEA Frontier Visualization and Diagnostic Plots
This program imports functions from the main Stage 2 regression script
and creates visualizations using the data and results directly.

Key Visualizations:
1. DEA frontier distribution (efficiency scores)
2. Efficiency trends over time
3. Country-level rankings
4. Governance vs. FPE relationship
5. Bootstrap results (if available)
6. Diagnostic plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import from the main script
from High_Income_countries_test import (
    build_stage2_panel,
    build_design_matrix,
    fit_truncated_regression,
    parametric_bootstrap,
    N_BOOTSTRAP,
    GDP_THRESH  # Import the threshold constant
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

# =============================================================================
# VISUALIZATION 1: FPE DISTRIBUTION AND FRONTIER
# =============================================================================

def plot_fpe_distribution(panel_full, panel_restr, save=True):
    """Plot FPE score distributions with frontier highlighted"""
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

    # 2. Restricted sample distribution
    ax2 = axes[0, 1]
    frontier_restr = panel_restr[panel_restr['fpe_score'] >= 0.9999]
    non_frontier_restr = panel_restr[panel_restr['fpe_score'] < 0.9999]

    ax2.hist(non_frontier_restr['fpe_score'], bins=30, alpha=0.7,
             color='seagreen', label='Inefficient DMUs', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label=f'Frontier (n={len(frontier_restr)})')
    ax2.set_xlabel('FPE Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Restricted Sample (GDP ≥ ${GDP_THRESH:,}): FPE Distribution\n(n={len(panel_restr)})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Kernel density comparison
    ax3 = axes[1, 0]
    if len(non_frontier_full) > 0:
        sns.kdeplot(non_frontier_full['fpe_score'], label='Full Sample',
                   ax=ax3, linewidth=2.5)
    if len(non_frontier_restr) > 0:
        sns.kdeplot(non_frontier_restr['fpe_score'], label='Restricted Sample',
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
    if len(non_frontier_restr) > 0:
        data_to_plot.append(non_frontier_restr['fpe_score'].values)
        labels.append('Restricted (non-frontier)')

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
        plt.savefig('figures/fpe_distribution.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/fpe_distribution.png")
    plt.show()
    return fig

# =============================================================================
# VISUALIZATION 2: TRENDS OVER TIME
# =============================================================================

def plot_time_trends(panel_full, panel_restr, save=True):
    """Plot FPE trends over time"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full sample - mean FPE by year
    ax1 = axes[0, 0]
    yearly_full = panel_full.groupby('year').agg({
        'fpe_score': ['mean', 'std', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    yearly_full.columns = ['year', 'mean_fpe', 'std_fpe', 'n_obs', 'n_frontier']

    ax1.plot(yearly_full['year'], yearly_full['mean_fpe'],
             marker='o', linewidth=2.5, color='steelblue', label='Mean FPE')
    ax1.fill_between(yearly_full['year'],
                     yearly_full['mean_fpe'] - yearly_full['std_fpe'],
                     yearly_full['mean_fpe'] + yearly_full['std_fpe'],
                     alpha=0.2, color='steelblue')

    # Add frontier count as secondary axis
    ax1_2 = ax1.twinx()
    ax1_2.bar(yearly_full['year'], yearly_full['n_frontier'],
             alpha=0.3, color='coral', label='Frontier Count')
    ax1_2.set_ylabel('Number of Frontier DMUs', fontsize=11, color='coral')
    ax1_2.tick_params(axis='y', labelcolor='coral')

    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Mean FPE Score', fontsize=11, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_title('Full Sample: FPE Trends Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Restricted sample trends
    ax2 = axes[0, 1]
    yearly_restr = panel_restr.groupby('year').agg({
        'fpe_score': ['mean', 'std', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    yearly_restr.columns = ['year', 'mean_fpe', 'std_fpe', 'n_obs', 'n_frontier']

    ax2.plot(yearly_restr['year'], yearly_restr['mean_fpe'],
             marker='s', linewidth=2.5, color='seagreen', label='Mean FPE')
    ax2.fill_between(yearly_restr['year'],
                     yearly_restr['mean_fpe'] - yearly_restr['std_fpe'],
                     yearly_restr['mean_fpe'] + yearly_restr['std_fpe'],
                     alpha=0.2, color='seagreen')

    ax2_2 = ax2.twinx()
    ax2_2.bar(yearly_restr['year'], yearly_restr['n_frontier'],
             alpha=0.3, color='coral', label='Frontier Count')
    ax2_2.set_ylabel('Number of Frontier DMUs', fontsize=11, color='coral')
    ax2_2.tick_params(axis='y', labelcolor='coral')

    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Mean FPE Score', fontsize=11, color='seagreen')
    ax2.tick_params(axis='y', labelcolor='seagreen')
    ax2.set_title('Restricted Sample: FPE Trends Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Frontier proportion over time
    ax3 = axes[1, 0]
    full_frontier_pct = (yearly_full['n_frontier'] / yearly_full['n_obs'] * 100)
    restr_frontier_pct = (yearly_restr['n_frontier'] / yearly_restr['n_obs'] * 100)

    ax3.plot(yearly_full['year'], full_frontier_pct,
             marker='o', linewidth=2.5, color='steelblue', label='Full Sample')
    ax3.plot(yearly_restr['year'], restr_frontier_pct,
             marker='s', linewidth=2.5, color='seagreen', label='Restricted Sample')
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Frontier DMUs (%)', fontsize=11)
    ax3.set_title('Proportion of Frontier DMUs Over Time', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Year-over-year change
    ax4 = axes[1, 1]
    full_change = yearly_full['mean_fpe'].diff()
    restr_change = yearly_restr['mean_fpe'].diff()

    ax4.bar(yearly_full['year'][1:], full_change[1:],
            alpha=0.6, color='steelblue', label='Full Sample', width=0.35, align='center')
    ax4.bar(yearly_restr['year'][1:] + 0.35, restr_change[1:],
            alpha=0.6, color='seagreen', label='Restricted Sample', width=0.35, align='center')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Change in Mean FPE', fontsize=11)
    ax4.set_title('Year-over-Year Changes in Mean FPE', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/fpe_time_trends.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/fpe_time_trends.png")
    plt.show()
    return fig

# =============================================================================
# VISUALIZATION 3: COUNTRY RANKINGS
# =============================================================================

def plot_country_rankings(panel_full, save=True):
    """Plot country rankings by average FPE score"""
    # Calculate average FPE by country
    country_avg = panel_full.groupby('iso3').agg({
        'fpe_score': ['mean', 'std', 'count'],
        'is_frontier': lambda x: (x >= 0.9999).sum()
    }).reset_index()
    country_avg.columns = ['iso3', 'mean_fpe', 'std_fpe', 'n_obs', 'n_frontier']
    country_avg = country_avg.sort_values('mean_fpe', ascending=False)
    country_avg['frontier_pct'] = (country_avg['n_frontier'] / country_avg['n_obs'] * 100)

    # Top 20 and bottom 20
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Top 20
    ax1 = axes[0]
    top20 = country_avg.head(20)
    colors = ['green' if x > 0.8 else 'steelblue' for x in top20['mean_fpe']]
    bars1 = ax1.barh(top20['iso3'], top20['mean_fpe'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Average FPE Score', fontsize=12)
    ax1.set_title(f'Top 20 Countries by Average FPE Score\n(2000-2021)', fontsize=12)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')

    # Add error bars
    ax1.errorbar(top20['mean_fpe'], range(len(top20)),
                xerr=top20['std_fpe'], fmt='none', color='gray', capsize=2, alpha=0.5)

    # Add frontier percentage annotations
    for i, (idx, row) in enumerate(top20.iterrows()):
        if row['frontier_pct'] > 0:
            ax1.text(row['mean_fpe'] + 0.01, i, f' {row["frontier_pct"]:.0f}% frontier',
                    va='center', fontsize=8, color='coral')

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom 20
    ax2 = axes[1]
    bottom20 = country_avg.tail(20).sort_values('mean_fpe', ascending=True)
    colors = ['coral' if x < 0.5 else 'steelblue' for x in bottom20['mean_fpe']]
    bars2 = ax2.barh(bottom20['iso3'], bottom20['mean_fpe'], color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Average FPE Score', fontsize=12)
    ax2.set_title(f'Bottom 20 Countries by Average FPE Score\n(2000-2021)', fontsize=12)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Frontier')

    # Add error bars
    ax2.errorbar(bottom20['mean_fpe'], range(len(bottom20)),
                xerr=bottom20['std_fpe'], fmt='none', color='gray', capsize=2, alpha=0.5)

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/country_rankings.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/country_rankings.png")
    plt.show()

    # Return for potential further analysis
    return country_avg

# =============================================================================
# VISUALIZATION 4: GOVERNANCE vs FPE RELATIONSHIP
# =============================================================================

def plot_governance_relationship(panel_full, panel_restr, save=True):
    """Plot relationship between WGI composite and FPE scores"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full sample: WGI vs FPE
    ax1 = axes[0, 0]
    non_frontier_full = panel_full[panel_full['fpe_score'] < 0.9999]
    frontier_full = panel_full[panel_full['fpe_score'] >= 0.9999]

    ax1.scatter(non_frontier_full['wgi_composite'], non_frontier_full['fpe_score'],
               alpha=0.5, s=30, color='steelblue', label='Inefficient DMUs')
    ax1.scatter(frontier_full['wgi_composite'], frontier_full['fpe_score'],
               alpha=0.7, s=80, color='red', marker='*', label='Frontier DMUs')

    # Add trend line for non-frontier
    if len(non_frontier_full) > 1:
        z = np.polyfit(non_frontier_full['wgi_composite'], non_frontier_full['fpe_score'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(non_frontier_full['wgi_composite'].min(),
                            non_frontier_full['wgi_composite'].max(), 100)
        ax1.plot(x_trend, p(x_trend), '--', color='steelblue', alpha=0.5,
                label=f'Trend: slope={z[0]:.3f}')

    ax1.set_xlabel('WGI Composite (Governance)', fontsize=11)
    ax1.set_ylabel('FPE Score', fontsize=11)
    ax1.set_title('Full Sample: Governance vs Financial Protection', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Restricted sample: WGI vs FPE
    ax2 = axes[0, 1]
    non_frontier_restr = panel_restr[panel_restr['fpe_score'] < 0.9999]
    frontier_restr = panel_restr[panel_restr['fpe_score'] >= 0.9999]

    ax2.scatter(non_frontier_restr['wgi_composite'], non_frontier_restr['fpe_score'],
               alpha=0.5, s=30, color='seagreen', label='Inefficient DMUs')
    ax2.scatter(frontier_restr['wgi_composite'], frontier_restr['fpe_score'],
               alpha=0.7, s=80, color='red', marker='*', label='Frontier DMUs')

    if len(non_frontier_restr) > 1:
        z = np.polyfit(non_frontier_restr['wgi_composite'], non_frontier_restr['fpe_score'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(non_frontier_restr['wgi_composite'].min(),
                            non_frontier_restr['wgi_composite'].max(), 100)
        ax2.plot(x_trend, p(x_trend), '--', color='seagreen', alpha=0.5,
                label=f'Trend: slope={z[0]:.3f}')

    ax2.set_xlabel('WGI Composite (Governance)', fontsize=11)
    ax2.set_ylabel('FPE Score', fontsize=11)
    ax2.set_title('Restricted Sample: Governance vs Financial Protection', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. GINI vs FPE
    ax3 = axes[1, 0]
    if 'gini' in panel_full.columns and not panel_full['gini'].isna().all():
        # Remove NaN for GINI
        gini_full = panel_full.dropna(subset=['gini'])
        gini_frontier = gini_full[gini_full['fpe_score'] >= 0.9999]
        gini_non_frontier = gini_full[gini_full['fpe_score'] < 0.9999]

        ax3.scatter(gini_non_frontier['gini'], gini_non_frontier['fpe_score'],
                   alpha=0.5, s=30, color='steelblue', label='Inefficient DMUs')
        ax3.scatter(gini_frontier['gini'], gini_frontier['fpe_score'],
                   alpha=0.7, s=80, color='red', marker='*', label='Frontier DMUs')

        if len(gini_non_frontier) > 1:
            z = np.polyfit(gini_non_frontier['gini'], gini_non_frontier['fpe_score'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(gini_non_frontier['gini'].min(),
                                gini_non_frontier['gini'].max(), 100)
            ax3.plot(x_trend, p(x_trend), '--', color='steelblue', alpha=0.5,
                    label=f'Trend: slope={z[0]:.3f}')

        ax3.set_xlabel('GINI Coefficient', fontsize=11)
        ax3.set_ylabel('FPE Score', fontsize=11)
        ax3.set_title('Full Sample: Inequality vs Financial Protection', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Heatmap of WGI vs FPE by region (if region data available)
    ax4 = axes[1, 1]
    # We'll use a 2D histogram as a proxy for heatmap
    if len(non_frontier_full) > 0:
        h = ax4.hist2d(non_frontier_full['wgi_composite'], non_frontier_full['fpe_score'],
                      bins=20, cmap='YlOrRd', alpha=0.7)
        plt.colorbar(h[3], ax=ax4, label='Frequency')
        ax4.scatter(frontier_full['wgi_composite'], frontier_full['fpe_score'],
                   alpha=0.7, s=80, color='blue', marker='*', label='Frontier')
        ax4.set_xlabel('WGI Composite (Governance)', fontsize=11)
        ax4.set_ylabel('FPE Score', fontsize=11)
        ax4.set_title('Density of Observations: WGI vs FPE', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/governance_relationship.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/governance_relationship.png")
    plt.show()
    return fig

# =============================================================================
# VISUALIZATION 5: BOOTSTRAP RESULTS (if available)
# =============================================================================

def plot_bootstrap_results(panel_full, save=True):
    """Run a quick bootstrap and visualize the results"""
    print("\n" + "="*65)
    print("  BOOTSTRAP DIAGNOSTIC PLOTS")
    print("="*65)

    # Prepare data for truncated regression
    df_trunc = panel_full[panel_full['fpe_score'] < 0.9999].copy()
    if len(df_trunc) < 10:
        print("  Too few non-frontier observations for bootstrap visualization")
        return None

    # Use the imported functions
    y = df_trunc['fpe_score'].values
    X, col_names = build_design_matrix(df_trunc)

    print(f"  Fitting truncated regression on {len(df_trunc)} observations...")
    beta_hat, sigma_hat, res = fit_truncated_regression(y, X)

    print(f"  Running bootstrap (B=500 for visualization)...")
    # Use smaller B for visualization
    boot_betas = parametric_bootstrap(y, X, beta_hat, sigma_hat, n_bootstrap=500)

    if len(boot_betas) < 10:
        print("  Not enough successful bootstrap draws for visualization")
        return None

    # Create bootstrap visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot coefficient distributions
    for i, name in enumerate(['const', 'wgi', 'gini']):
        if name in col_names[:3]:
            idx = col_names.index(name)
            ax = axes[i//3, i%3] if i < 3 else axes[1, i-3]

            # Histogram of bootstrap coefficients
            ax.hist(boot_betas[:, idx], bins=30, alpha=0.7, color='steelblue',
                   edgecolor='black', linewidth=0.5)
            ax.axvline(beta_hat[idx], color='red', linestyle='--', linewidth=2,
                      label=f'Estimate: {beta_hat[idx]:.3f}')
            ax.axvline(np.percentile(boot_betas[:, idx], 2.5), color='gray',
                      linestyle=':', linewidth=1.5, label='95% CI')
            ax.axvline(np.percentile(boot_betas[:, idx], 97.5), color='gray',
                      linestyle=':', linewidth=1.5)

            ax.set_xlabel(f'{name} Coefficient', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Bootstrap Distribution: {name}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    # 6th subplot: Correlation between coefficients
    ax = axes[1, 2]
    if 'wgi' in col_names[:3] and 'gini' in col_names[:3]:
        wgi_idx = col_names.index('wgi')
        gini_idx = col_names.index('gini')
        ax.scatter(boot_betas[:, wgi_idx], boot_betas[:, gini_idx],
                  alpha=0.5, s=20, color='coral')
        ax.axvline(beta_hat[wgi_idx], color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(beta_hat[gini_idx], color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('WGI Coefficient', fontsize=10)
        ax.set_ylabel('GINI Coefficient', fontsize=10)
        ax.set_title('Bootstrap Coef Correlation', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/bootstrap_results.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/bootstrap_results.png")
    plt.show()

    return boot_betas

# =============================================================================
# VISUALIZATION 6: DIAGNOSTIC PLOTS (NEW)
# =============================================================================

def plot_diagnostic_plots(panel_full, save=True):
    """Create diagnostic plots for the truncated regression"""
    print("\n" + "="*65)
    print("  DIAGNOSTIC PLOTS")
    print("="*65)

    # Prepare data
    df_trunc = panel_full[panel_full['fpe_score'] < 0.9999].copy()
    if len(df_trunc) < 10:
        print("  Too few non-frontier observations")
        return None

    # Fit model
    y = df_trunc['fpe_score'].values
    X, col_names = build_design_matrix(df_trunc)
    beta_hat, sigma_hat, res = fit_truncated_regression(y, X)

    # Calculate predictions and residuals
    mu_hat = X @ beta_hat
    residuals = y - mu_hat
    standardized_residuals = residuals / sigma_hat

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(mu_hat, residuals, alpha=0.5, s=30, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Fitted Values', fontsize=11)
    ax1.set_ylabel('Residuals', fontsize=11)
    ax1.set_title('Residuals vs Fitted Values', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Q-Q plot
    ax2 = axes[0, 1]
    stats.probplot(standardized_residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Standardized Residuals)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Histogram of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.5, density=True)
    # Overlay normal distribution
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    ax3.plot(x_norm, stats.norm.pdf(x_norm, 0, residuals.std()),
            'r-', linewidth=2, label='Normal Fit')
    ax3.set_xlabel('Residuals', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Residual Distribution', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Scale-Location plot
    ax4 = axes[1, 1]
    sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
    ax4.scatter(mu_hat, sqrt_abs_resid, alpha=0.5, s=30, color='steelblue')
    # Add trend line
    if len(mu_hat) > 1:
        z = np.polyfit(mu_hat, sqrt_abs_resid, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(mu_hat.min(), mu_hat.max(), 100)
        ax4.plot(x_trend, p(x_trend), 'r-', linewidth=1.5)
    ax4.set_xlabel('Fitted Values', fontsize=11)
    ax4.set_ylabel('√|Standardized Residuals|', fontsize=11)
    ax4.set_title('Scale-Location Plot', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('figures/diagnostic_plots.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: figures/diagnostic_plots.png")
    plt.show()

    return mu_hat, residuals, standardized_residuals

# =============================================================================
# MAIN VISUALIZATION PIPELINE
# =============================================================================

def create_all_visualizations():
    """Run all visualizations"""
    print("\n" + "="*65)
    print("  DEA FRONTIER VISUALIZATION PIPELINE")
    print("="*65)

    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)

    # Load data using imported function
    print("\n  Loading data from main program...")
    panel_full, panel_restr = build_stage2_panel()

    # Add frontier flag for easier plotting
    panel_full['is_frontier'] = panel_full['fpe_score'] >= 0.9999
    panel_restr['is_frontier'] = panel_restr['fpe_score'] >= 0.9999

    print(f"\n  Data loaded:")
    print(f"    Full sample: {len(panel_full)} observations")
    print(f"    Restricted sample: {len(panel_restr)} observations")

    # Generate all visualizations
    print("\n  Generating visualizations...")

    # 1. FPE distribution
    print("\n  1. FPE Distribution and Frontier")
    plot_fpe_distribution(panel_full, panel_restr, save=True)

    # 2. Time trends
    print("\n  2. Time Trends")
    plot_time_trends(panel_full, panel_restr, save=True)

    # 3. Country rankings
    print("\n  3. Country Rankings")
    country_avg = plot_country_rankings(panel_full, save=True)

    # 4. Governance relationship
    print("\n  4. Governance vs FPE Relationship")
    plot_governance_relationship(panel_full, panel_restr, save=True)

    # 5. Bootstrap results (optional)
    print("\n  5. Bootstrap Diagnostic Plots")
    try:
        plot_bootstrap_results(panel_full, save=True)
    except Exception as e:
        print(f"    Bootstrap visualization skipped: {str(e)}")

    # 6. Diagnostic plots
    print("\n  6. Regression Diagnostic Plots")
    try:
        plot_diagnostic_plots(panel_full, save=True)
    except Exception as e:
        print(f"    Diagnostic plots skipped: {str(e)}")

    print("\n" + "="*65)
    print("  ✅ ALL VISUALIZATIONS COMPLETE")
    print(f"  Visualizations saved to: figures/")
    print("="*65)

    return country_avg

# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == '__main__':
    # Run the visualization pipeline
    country_rankings = create_all_visualizations()

    # Optionally save country rankings to CSV
    if country_rankings is not None:
        country_rankings.to_csv('figures/country_fpe_rankings.csv', index=False)
        print(f"  Country rankings saved to: figures/country_fpe_rankings.csv")