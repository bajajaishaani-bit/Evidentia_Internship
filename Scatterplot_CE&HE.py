#Catastrophic Expenditure and Health Expenditure (as a % of GDP)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file1 = os.path.join(script_dir, '/Users/aishaanibajaj/Downloads/CHE10_cleaned.csv')
file2 = os.path.join(script_dir, '/Users/aishaanibajaj/Downloads/OOP_cleaned.csv')

# Check if files exist
if not os.path.exists(file1):
    print(f"Error: Cannot find {file1}")
    print(f"Current working directory: {os.getcwd()}")
    print("\nPlease update the file paths in the script.")
    print("You can either:")
    print("1. Move the CSV files to the same directory as this script")
    print("2. Update the file paths below with the correct locations")
    exit(1)

try:
    # Load the data
    df_health = pd.read_csv(file1)
    df_oop = pd.read_csv(file2)
    print("Files loaded successfully!")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("\nPlease make sure the CSV files are in the correct location.")
    print(f"Looking for files at: {file1}")
    print(f"and: {file2}")
    exit(1)

# Continue with the analysis
print("\nProcessing data...")

# Filter for the specific indicators
health_spending_df = df_health[
    df_health['Indicator Name'] == 'Population with health spending >10% of household budget (%)']
oop_df = df_oop[df_oop['Indicator Name'] == 'Out-of-pocket expenditure (% of current health expenditure)']

# Get country codes for merging
health_spending_df = health_spending_df[['Country Code', 'Country Name'] + [str(year) for year in range(2000, 2025)]]
oop_df = oop_df[['Country Code', 'Country Name'] + [str(year) for year in range(2000, 2025)]]

# Prepare data for scatter plot (using the most recent common year with data)
# Let's use data from 2019 (pre-pandemic) for better comparison
year = '2019'

# Merge the two datasets on Country Code
merged = pd.merge(
    health_spending_df[['Country Code', 'Country Name', year]],
    oop_df[['Country Code', year]],
    on='Country Code',
    suffixes=('_health', '_oop')
)

# Rename columns
merged.columns = ['Country Code', 'Country Name', 'Health_Spending_10pct', 'OOP_pct']

# Remove missing values
merged_clean = merged.dropna(subset=['Health_Spending_10pct', 'OOP_pct'])

if len(merged_clean) == 0:
    print("No data available for 2019. Trying to find the most recent year with data...")

    # Find the most recent year with data
    years_with_data = []
    for y in range(2000, 2025):
        year_str = str(y)
        temp_merged = pd.merge(
            health_spending_df[['Country Code', 'Country Name', year_str]],
            oop_df[['Country Code', year_str]],
            on='Country Code',
            suffixes=('_health', '_oop')
        )
        temp_merged.columns = ['Country Code', 'Country Name', 'Health_Spending_10pct', 'OOP_pct']
        temp_clean = temp_merged.dropna()
        if len(temp_clean) > 0:
            years_with_data.append((y, len(temp_clean)))

    if years_with_data:
        # Use the year with the most data
        best_year = max(years_with_data, key=lambda x: x[1])[0]
        year = str(best_year)
        print(f"Using data from {year}")

        merged = pd.merge(
            health_spending_df[['Country Code', 'Country Name', year]],
            oop_df[['Country Code', year]],
            on='Country Code',
            suffixes=('_health', '_oop')
        )
        merged.columns = ['Country Code', 'Country Name', 'Health_Spending_10pct', 'OOP_pct']
        merged_clean = merged.dropna()
    else:
        print("No data available for any year. Please check the CSV files.")
        exit(1)

# Also let's prepare data for all years combined (time series scatter)
all_data = []
for year in range(2000, 2025):
    year_str = str(year)
    if year_str in health_spending_df.columns and year_str in oop_df.columns:
        temp_merged = pd.merge(
            health_spending_df[['Country Code', 'Country Name', year_str]],
            oop_df[['Country Code', year_str]],
            on='Country Code',
            suffixes=('_health', '_oop')
        )
        temp_merged.columns = ['Country Code', 'Country Name', 'Health_Spending_10pct', 'OOP_pct']
        temp_merged['Year'] = year
        all_data.append(temp_merged.dropna())

# Combine all years
all_years_data = pd.concat(all_data, ignore_index=True)

if len(all_years_data) == 0:
    print("No data available for any year. Please check the CSV files.")
    exit(1)

print(f"Found data for {len(all_years_data)} country-year observations")
print(f"Found data for {len(merged_clean)} countries in {year}")

# Create the scatterplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Most recent data (2019)
ax1 = axes[0]
scatter1 = ax1.scatter(merged_clean['OOP_pct'], merged_clean['Health_Spending_10pct'],
                       alpha=0.6, s=80, c='steelblue', edgecolors='black', linewidth=0.5)

# Add trend line
if len(merged_clean) > 1:
    z = np.polyfit(merged_clean['OOP_pct'], merged_clean['Health_Spending_10pct'], 1)
    p = np.poly1d(z)
    ax1.plot(np.sort(merged_clean['OOP_pct']),
             p(np.sort(merged_clean['OOP_pct'])),
             "r--", alpha=0.8,
             label=f'Trend line (r={merged_clean["OOP_pct"].corr(merged_clean["Health_Spending_10pct"]):.2f})')

# Calculate correlation
correlation = merged_clean['OOP_pct'].corr(merged_clean['Health_Spending_10pct'])
ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_xlabel('Out-of-pocket expenditure (% of current health expenditure)', fontsize=11)
ax1.set_ylabel('Population with health spending >10% of household budget (%)', fontsize=11)
n_countries = len(merged_clean)
ax1.set_title(f'Health Spending Burden vs OOP Expenditure ({year})\n{n_countries} countries', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Highlight some interesting countries
highlight_countries = ['IND', 'CHN', 'USA', 'ARM', 'GEO', 'NGA']
for code in highlight_countries:
    country_data = merged_clean[merged_clean['Country Code'] == code]
    if not country_data.empty:
        ax1.annotate(country_data['Country Name'].values[0],
                     (country_data['OOP_pct'].values[0], country_data['Health_Spending_10pct'].values[0]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

# Plot 2: All years combined (with color mapping by year)
ax2 = axes[1]
scatter2 = ax2.scatter(all_years_data['OOP_pct'], all_years_data['Health_Spending_10pct'],
                       c=all_years_data['Year'], cmap='viridis', alpha=0.5, s=30,
                       edgecolors='none')

# Add colorbar
cbar = plt.colorbar(scatter2, ax=ax2)
cbar.set_label('Year', fontsize=10)

# Add overall trend line
if len(all_years_data) > 1:
    z2 = np.polyfit(all_years_data['OOP_pct'], all_years_data['Health_Spending_10pct'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(np.sort(all_years_data['OOP_pct']),
             p2(np.sort(all_years_data['OOP_pct'])),
             "r--", alpha=0.8, label='Trend line')

# Calculate overall correlation
correlation_all = all_years_data['OOP_pct'].corr(all_years_data['Health_Spending_10pct'])
ax2.text(0.05, 0.95, f'Overall Correlation: {correlation_all:.3f}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel('Out-of-pocket expenditure (% of current health expenditure)', fontsize=11)
ax2.set_ylabel('Population with health spending >10% of household budget (%)', fontsize=11)
ax2.set_title('Health Spending Burden vs OOP Expenditure\nAll Years (2000-2024)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# Print statistical summary
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(f"\nNumber of country-year observations: {len(all_years_data)}")
print(f"Number of countries in {year} analysis: {len(merged_clean)}")
print(f"\nCorrelation coefficient ({year}): {correlation:.3f}")
print(f"Correlation coefficient (all years): {correlation_all:.3f}")

# Show top 10 countries with highest health spending >10% in the selected year
top_10 = merged_clean.nlargest(10, 'Health_Spending_10pct')[['Country Name', 'Health_Spending_10pct', 'OOP_pct']]
print(f"\nTop 10 countries with highest % spending >10% of household budget ({year}):")
print(top_10.to_string(index=False))

# Calculate summary statistics
print(f"\nSummary Statistics ({year}):")
print(f"  OOP Expenditure - Mean: {merged_clean['OOP_pct'].mean():.2f}%, Std: {merged_clean['OOP_pct'].std():.2f}%")
print(
    f"  Health Spending >10% - Mean: {merged_clean['Health_Spending_10pct'].mean():.2f}%, Std: {merged_clean['Health_Spending_10pct'].std():.2f}%")

print("\nAnalysis complete!")