import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_data_with_prompt():
    """Load all datasets with user-provided paths"""

    print("=" * 80)
    print("DATA LOADER FOR HEALTH AND GOVERNANCE INDICATORS")
    print("=" * 80)

    files_info = {
        'CHE': {
            'description': 'Population with health spending >10% of household budget',
            'default': 'CHE10_cleaned.csv',
            'prompt': 'Enter path for CHE (health spending) file'
        },
        'GDP': {
            'description': 'GDP per capita',
            'default': 'GDP_perCapita_cleaned.csv',
            'prompt': 'Enter path for GDP per capita file'
        },
        'GINI': {
            'description': 'Gini index',
            'default': 'GINI_cleaned.csv',
            'prompt': 'Enter path for GINI file'
        },
        'WGI_GE': {
            'description': 'WGI - Government Effectiveness (GE)',
            'default': 'WGI_cleaned.csv',
            'prompt': 'Enter path for WGI file (will extract GE indicator)'
        },
        'OOP': {
            'description': 'Out-of-pocket expenditure',
            'default': 'OOP_cleaned.csv',
            'prompt': 'Enter path for OOP file'
        },
        'UHC': {
            'description': 'UHC Service Coverage Index',
            'default': 'UHC_cleaned.csv',
            'prompt': 'Enter path for UHC file'
        }
    }

    data = {}

    print("\nPlease provide the paths to each data file.")
    print("Press Enter to use default filename (assuming file is in current directory)\n")

    for key, info in files_info.items():
        print(f"\n--- {info['description']} ---")
        user_path = input(f"{info['prompt']} [{info['default']}]: ").strip()

        if not user_path:
            user_path = info['default']

        try:
            df = pd.read_csv(user_path)
            data[key] = df
            print(f"✓ Loaded: {user_path} ({len(df)} rows, {len(df.columns)} columns)")
        except FileNotFoundError:
            print(f"✗ File not found: {user_path}")
            retry = input(f"Try again? Enter new path or 'skip' to skip this file: ").strip()
            if retry.lower() != 'skip':
                try:
                    df = pd.read_csv(retry)
                    data[key] = df
                    print(f"✓ Loaded: {retry}")
                except:
                    print(f"✗ Could not load {retry}. Skipping this file.")
                    data[key] = None
            else:
                data[key] = None
                print(f"Skipping {key} file.")

    return data


def extract_yearly_country_data_standard(df, var_name):
    """Extract which countries have data for each year from standard dataframe"""
    if df is None:
        return {}

    # Find year columns (columns that are numeric or look like years)
    year_columns = []
    for col in df.columns:
        try:
            # Check if column name can be converted to int and is between 1990 and 2030
            year_int = int(col)
            if 1990 <= year_int <= 2030:
                year_columns.append(col)
        except (ValueError, TypeError):
            continue

    year_columns = sorted(year_columns, key=lambda x: int(x))

    result = {}

    for year in year_columns:
        # Get countries with non-null data for this year
        countries_with_data = df[df[year].notna()]['Country Name'].tolist()
        if countries_with_data:
            result[year] = {
                'count': len(countries_with_data),
                'countries': countries_with_data
            }

    return result


def extract_yearly_country_data_wgi_ge(df, var_name):
    """Extract which countries have data for each year from WGI - focusing only on GE (Government Effectiveness)"""
    if df is None:
        return {}

    result = {}

    # Find all columns that start with 'WGI.GE_' (Government Effectiveness)
    ge_columns = [col for col in df.columns if col.startswith('WGI.GE_')]

    if not ge_columns:
        print("  Warning: No WGI.GE_ columns found. Available WGI columns:")
        wgi_cols = [col for col in df.columns if col.startswith('WGI.')]
        print(f"  Found: {wgi_cols[:10]}...")
        return result

    for col in ge_columns:
        # Extract year from column name (e.g., 'WGI.GE_2000' -> '2000')
        year = col.split('_')[-1]

        # Get countries with non-null data for this year
        countries_with_data = df[df[col].notna()]['Country Name'].tolist()

        if countries_with_data:
            result[year] = {
                'count': len(countries_with_data),
                'countries': countries_with_data,
                'indicator': 'Government Effectiveness (GE)'
            }

    return result


def get_country_list_for_year(data_dict, var_key, year):
    """Get list of countries for a specific variable and year"""
    if var_key not in data_dict or data_dict[var_key] is None:
        return []

    df = data_dict[var_key]

    if var_key == 'WGI_GE':
        col_name = f'WGI.GE_{year}'
        if col_name not in df.columns:
            return []
        countries = df[df[col_name].notna()]['Country Name'].tolist()
        return sorted(countries)
    else:
        if str(year) not in df.columns:
            return []
        countries = df[df[str(year)].notna()]['Country Name'].tolist()
        return sorted(countries)


def analyze_all_variables(data):
    """Analyze data availability across all variables"""

    variable_names = {
        'CHE': 'Health spending >10%',
        'GDP': 'GDP per capita',
        'GINI': 'Gini index',
        'WGI_GE': 'WGI - Government Effectiveness (GE)',
        'OOP': 'Out-of-pocket expenditure',
        'UHC': 'UHC indicator'
    }

    all_analyses = {}
    all_years = set()

    print("\n" + "=" * 80)
    print("DATA AVAILABILITY ANALYSIS BY VARIABLE")
    print("=" * 80)

    for var_key, var_df in data.items():
        if var_df is None:
            print(f"\n⚠️ {variable_names.get(var_key, var_key)}: No data loaded")
            continue

        print(f"\n{'=' * 60}")
        print(f"VARIABLE: {variable_names.get(var_key, var_key)}")
        print(f"{'=' * 60}")

        if var_key == 'WGI_GE':
            yearly_data = extract_yearly_country_data_wgi_ge(var_df, var_key)
        else:
            yearly_data = extract_yearly_country_data_standard(var_df, var_key)

        all_analyses[var_key] = yearly_data

        for year in sorted(yearly_data.keys(), key=lambda x: int(x)):
            all_years.add(year)
            count = yearly_data[year]['count']
            countries = yearly_data[year]['countries']
            print(f"\n  📍 {year}: {count} countries")
            print(f"     Countries: {', '.join(countries)}")

    # Create comprehensive table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATA AVAILABILITY TABLE")
    print("(Number of countries with data for each variable by year)")
    print("=" * 80)

    # Sort years
    sorted_years = sorted(all_years, key=lambda x: int(x))

    # Create header
    available_vars = [v for v in ['CHE', 'GDP', 'GINI', 'WGI_GE', 'OOP', 'UHC'] if
                      v in all_analyses and all_analyses[v]]
    header = ["Year"] + [variable_names[v] for v in available_vars]

    # Print table
    col_widths = [8] + [30] * len(available_vars)
    header_line = ""
    for i, h in enumerate(header):
        header_line += f"{h:<{col_widths[i]}}"
    print("\n" + header_line)
    print("-" * sum(col_widths))

    for year in sorted_years:
        row = f"{year:<{col_widths[0]}}"
        for i, var_key in enumerate(available_vars):
            count = all_analyses[var_key].get(year, {}).get('count', 0)
            row += f"{count:<{col_widths[i + 1]}}"
        print(row)

    # Specific year output as requested (2018 example)
    print("\n" + "=" * 80)
    print("DETAILED YEAR-BY-YEAR COUNTRY LISTS")
    print("=" * 80)

    for year in sorted_years:
        print(f"\n{'=' * 60}")
        print(f"📅 YEAR: {year}")
        print(f"{'=' * 60}")
        for var_key in available_vars:
            if year in all_analyses[var_key]:
                count = all_analyses[var_key][year]['count']
                countries = all_analyses[var_key][year]['countries']
                print(f"\n  📊 {variable_names[var_key]}:")
                print(f"     Total: {count} countries")
                print(f"     Countries: {', '.join(countries)}")
            else:
                print(f"\n  📊 {variable_names[var_key]}: No data")

    # Find years with complete data
    print("\n" + "=" * 80)
    print("COMPLETE DATA ANALYSIS")
    print("=" * 80)

    complete_years = []
    for year in sorted_years:
        all_have_data = True
        min_count = float('inf')
        for var_key in available_vars:
            count = all_analyses[var_key].get(year, {}).get('count', 0)
            if count == 0:
                all_have_data = False
                break
            min_count = min(min_count, count)

        if all_have_data:
            complete_years.append((year, min_count))

    if complete_years:
        print("\n✅ Years with data for ALL variables:")
        print(f"{'Year':<10} {'Minimum countries across variables':<35}")
        print("-" * 45)
        for year, min_count in complete_years:
            print(f"{year:<10} {min_count:<35}")
            # Show which countries are common across all variables for this year
            common_countries = None
            for var_key in available_vars:
                countries_set = set(all_analyses[var_key][year]['countries'])
                if common_countries is None:
                    common_countries = countries_set
                else:
                    common_countries = common_countries.intersection(countries_set)
            if common_countries:
                print(
                    f"     Common countries ({len(common_countries)}): {', '.join(sorted(common_countries)[:10])}{'...' if len(common_countries) > 10 else ''}")
    else:
        print("\n⚠️ No year has data for ALL variables simultaneously.")

    # Years with maximum countries for each variable
    print("\n" + "=" * 80)
    print("MAXIMUM COVERAGE BY VARIABLE")
    print("=" * 80)

    for var_key in available_vars:
        if all_analyses[var_key]:
            best_year = max(all_analyses[var_key].items(), key=lambda x: x[1]['count'])
            print(f"\n📈 {variable_names[var_key]}:")
            print(f"   Best year: {best_year[0]} with {best_year[1]['count']} countries")

    # Overall best year (maximum sum of countries)
    print("\n" + "=" * 80)
    print("OVERALL BEST YEARS (Max total country coverage)")
    print("=" * 80)

    year_totals = []
    for year in sorted_years:
        total = sum(all_analyses[var_key].get(year, {}).get('count', 0) for var_key in available_vars)
        year_totals.append((year, total))

    year_totals.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Year':<10} {'Total country-data points':<25} {'Average per variable':<20}")
    print("-" * 55)
    for year, total in year_totals[:5]:
        avg = total / len(available_vars)
        print(f"{year:<10} {total:<25} {avg:.1f}")

    # Export option
    print("\n" + "=" * 80)
    export_choice = input("\nDo you want to export the detailed country lists to CSV files? (y/n): ").strip().lower()

    if export_choice == 'y':
        # Export detailed country lists for each year and variable
        for var_key, yearly_data in all_analyses.items():
            if not yearly_data:
                continue
            var_name = variable_names.get(var_key, var_key).replace(' ', '_').replace('>', '').replace('%', '').replace(
                '-', '')
            rows = []
            for year, info in yearly_data.items():
                rows.append({
                    'year': year,
                    'country_count': info['count'],
                    'countries': '; '.join(info['countries'])
                })
            if rows:
                var_df = pd.DataFrame(rows)
                var_df.to_csv(f'{var_name}_yearly_countries.csv', index=False)
                print(f"✓ Exported: {var_name}_yearly_countries.csv")

        # Export summary matrix
        matrix_rows = []
        for year in sorted_years:
            row = {'year': year}
            for var_key in available_vars:
                row[variable_names[var_key]] = all_analyses[var_key].get(year, {}).get('count', 0)
            matrix_rows.append(row)
        matrix_df = pd.DataFrame(matrix_rows)
        matrix_df.to_csv('data_availability_summary.csv', index=False)
        print(f"✓ Exported: data_availability_summary.csv")

        # Export common countries for complete years
        if complete_years:
            common_rows = []
            for year, min_count in complete_years:
                common_countries = None
                for var_key in available_vars:
                    countries_set = set(all_analyses[var_key][year]['countries'])
                    if common_countries is None:
                        common_countries = countries_set
                    else:
                        common_countries = common_countries.intersection(countries_set)
                common_rows.append({
                    'year': year,
                    'total_countries_with_all_data': len(common_countries),
                    'countries': '; '.join(sorted(common_countries))
                })
            common_df = pd.DataFrame(common_rows)
            common_df.to_csv('common_countries_complete_years.csv', index=False)
            print(f"✓ Exported: common_countries_complete_years.csv")

    return all_analyses


def main():
    """Main function to run the analysis"""
    print("\n" + "=" * 80)
    print("HEALTH AND GOVERNANCE INDICATORS DATA ANALYZER")
    print("Focused on WGI - Government Effectiveness (GE)")
    print("=" * 80)

    # Load all data
    data = load_data_with_prompt()

    # Check if any data was loaded
    if all(df is None for df in data.values()):
        print("\nNo data files were loaded. Exiting.")
        return

    # Run analysis
    analyses = analyze_all_variables(data)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()