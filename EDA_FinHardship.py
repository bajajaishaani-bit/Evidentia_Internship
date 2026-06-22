#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import numpy as np


def load_data(file_path):
    """Load data from CSV or Excel (including multi-sheet Excel files)"""

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        print("Loading CSV file...")
        return pd.read_csv(file_path)

    elif file_extension in ['.xlsx', '.xls']:
        print("Loading Excel file...")

        try:
            # First, try to get sheet names without loading everything
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            sheets = excel_file.sheet_names

            print(f"Found {len(sheets)} sheets: {', '.join(sheets[:5])}")
            if len(sheets) > 5:
                print(f"  ... and {len(sheets) - 5} more sheets")

            print("\nSince this appears to be a WHO file with multiple sheets,")
            print("I'll examine each sheet and combine them for analysis.\n")

            # Store dataframes for each sheet
            all_data = []
            skipped_sheets = []

            for i, sheet in enumerate(sheets, 1):
                try:
                    print(f"Reading sheet {i}/{len(sheets)}: '{sheet}'...", end=' ')
                    df_sheet = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')

                    if len(df_sheet) > 0:  # Only add non-empty sheets
                        # Add source sheet column
                        df_sheet['source_sheet'] = sheet
                        all_data.append(df_sheet)
                        print(f"✓ {len(df_sheet)} rows, {len(df_sheet.columns)} columns")
                    else:
                        print(f"⚠ Empty sheet, skipping")
                        skipped_sheets.append(sheet)

                except Exception as e:
                    print(f"✗ Error reading sheet: {str(e)[:50]}")
                    skipped_sheets.append(sheet)

            if not all_data:
                print("\n❌ No valid data found in any sheet!")
                sys.exit(1)

            # Combine all valid sheets
            print(f"\nCombining {len(all_data)} sheets with data...")
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✓ Total combined dataset: {len(combined_df)} rows × {len(combined_df.columns)} columns")

            if skipped_sheets:
                print(f"⚠ Skipped {len(skipped_sheets)} empty/unreadable sheets")

            return combined_df

        except Exception as e:
            print(f"Error reading Excel file: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure you have installed: pip install openpyxl")
            print("2. The file might be corrupted or in an older Excel format")
            sys.exit(1)

    else:
        print(f"Error: Unsupported file type. Please provide a .csv, .xlsx, or .xls file.")
        sys.exit(1)


def perform_eda(df, file_name):
    """Perform EDA to answer the three questions"""

    print("\n" + "=" * 60)
    print("EDA RESULTS")
    print("=" * 60)

    # Display basic info
    print(f"\nFile analyzed: {file_name}")
    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn names: {list(df.columns)}")

    # Display first few rows to understand data structure
    print("\n" + "-" * 60)
    print("PREVIEW OF FIRST 5 ROWS:")
    print("-" * 60)
    print(df.head())

    # Question 1: Which year has the most observations?
    print("\n" + "-" * 60)
    print("QUESTION 1: Which year has the most observations?")
    print("-" * 60)

    # Try different approaches to find year information
    year_found = False

    # Look for year columns
    year_columns = [col for col in df.columns if 'year' in str(col).lower() or 'yr' in str(col).lower()]

    if year_columns:
        for year_col in year_columns:
            try:
                # Clean the year column
                year_series = pd.to_numeric(df[year_col], errors='coerce')
                valid_years = year_series.dropna()

                if len(valid_years) > 0:
                    year_counts = valid_years.value_counts().sort_index()
                    print(f"\nUsing column: '{year_col}'")
                    print(f"Observations per year (showing years with data):")
                    # Show all years that have data
                    for year, count in year_counts.items():
                        if pd.notna(year):
                            print(f"  {int(year)}: {count:,} observations")

                    max_year = year_counts.idxmax()
                    max_count = year_counts.max()
                    print(f"\n✓ Year with most observations: {int(max_year)} ({max_count:,} observations)")
                    year_found = True
                    break
            except:
                continue

    if not year_found:
        # Look for date columns
        date_columns = [col for col in df.columns if 'date' in str(col).lower()]
        if date_columns:
            for date_col in date_columns:
                try:
                    print(f"\nTrying to extract year from date column: '{date_col}'")
                    df['extracted_year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
                    year_counts = df['extracted_year'].value_counts().sort_index()

                    if len(year_counts) > 0:
                        print(f"\nObservations per year (showing years with data):")
                        for year, count in year_counts.items():
                            if pd.notna(year):
                                print(f"  {int(year)}: {count:,} observations")

                        max_year = year_counts.idxmax()
                        max_count = year_counts.max()
                        if pd.notna(max_year):
                            print(f"\n✓ Year with most observations: {int(max_year)} ({max_count:,} observations)")
                            year_found = True
                            break
                except:
                    continue

    if not year_found:
        print("⚠ Could not find year or date information in the data.")
        print("Available columns:", list(df.columns))

    # Question 2: How many countries are we looking at?
    print("\n" + "-" * 60)
    print("QUESTION 2: How many countries are we looking at?")
    print("-" * 60)

    # Common patterns for country columns
    country_patterns = ['country', 'nation', 'region', 'location', 'area', 'state', 'spatialdimension']
    country_columns = [col for col in df.columns
                       if any(pattern in str(col).lower() for pattern in country_patterns)]

    countries_found = False

    if country_columns:
        for country_col in country_columns:
            # Drop NaN values and get unique countries
            unique_countries = df[country_col].dropna().unique()

            if len(unique_countries) > 0 and len(unique_countries) < 500:  # Reasonable number of countries
                print(f"Using column: '{country_col}'")
                print(f"✓ Number of unique countries/regions: {len(unique_countries)}")

                # Clean and sort the country list (convert all to string, remove NaN)
                clean_countries = []
                for c in unique_countries:
                    if pd.notna(c) and str(c).strip() != '':
                        clean_countries.append(str(c))

                if clean_countries:
                    print(f"\nList of countries/regions:")
                    for i, country in enumerate(sorted(clean_countries), 1):
                        print(f"  {i}. {country}")
                    countries_found = True
                    break
    else:
        print("⚠ Could not find a country column.")
        print("Available columns:", list(df.columns))
        print("\nIf your data contains country information under a different column name,")
        print("please check the column list above.")

    if not countries_found:
        print("\n⚠ No valid country data found. Checking for other geographic columns...")
        # Look for any column that might contain geographic information
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if 1 < len(unique_vals) < 300:  # Could be country-like
                print(f"  Found column '{col}' with {len(unique_vals)} unique values")
                print(f"  Sample values: {list(unique_vals[:5])}")

    # Question 3: How many countries overlap?
    print("\n" + "-" * 60)
    print("QUESTION 3: How many countries overlap?")
    print("-" * 60)

    if countries_found and country_columns:
        country_col = country_columns[0]

        # Check if there are multiple years or categories to check overlap
        if year_found:
            # Find a year column to use
            if 'YEAR' in df.columns:
                year_col = 'YEAR'
            elif 'extracted_year' in df.columns:
                year_col = 'extracted_year'
            else:
                year_col = None

            if year_col and year_col in df.columns:
                # Clean the data - remove rows with missing country or year
                clean_df = df[[country_col, year_col]].dropna()
                clean_df = clean_df[clean_df[country_col].astype(str).str.strip() != '']

                if len(clean_df) > 0:
                    # Count countries per year
                    countries_per_year = clean_df.groupby(year_col)[country_col].nunique()
                    print(f"\nCountries present per year:")
                    for year in sorted(countries_per_year.index):
                        if pd.notna(year):
                            print(f"  {int(year)}: {countries_per_year[year]} countries")

                    # Find countries that appear in all years
                    all_years = clean_df[year_col].unique()
                    if len(all_years) > 1:
                        # Convert to strings for consistent comparison
                        countries_in_all_years = set(
                            clean_df[clean_df[country_col].astype(str)][country_col].astype(str).unique())

                        for year in all_years:
                            countries_in_year = set(
                                clean_df[clean_df[year_col] == year][country_col].astype(str).unique())
                            countries_in_all_years = countries_in_all_years.intersection(countries_in_year)

                        print(f"\n✓ Countries that appear in ALL {len(all_years)} years: {len(countries_in_all_years)}")
                        if len(countries_in_all_years) > 0 and len(countries_in_all_years) <= 20:
                            print(f"  These countries are: {sorted(countries_in_all_years)}")

                        # Countries that appear in at least 2 years
                        country_year_count = clean_df.groupby(clean_df[country_col].astype(str))[year_col].nunique()
                        overlapping_countries = country_year_count[country_year_count >= 2]
                        print(f"✓ Countries that appear in at least 2 different years: {len(overlapping_countries)}")

                        # Show the overlapping countries if not too many
                        if len(overlapping_countries) > 0 and len(overlapping_countries) <= 30:
                            print(f"\n  These countries are: {sorted(overlapping_countries.index.tolist())}")
                    else:
                        print("Only one year found in the data. Cannot check overlaps across different years.")
                else:
                    print("No valid country-year pairs found after cleaning data.")
            else:
                print("No year column found to check country overlap across time.")
        else:
            print("No year information found to check country overlap across time.")

        # Also check for overlap across different indicator types or other dimensions
        other_dimensions = ['INDICATOR_FULL', 'IndicatorCode', 'IND_NAME', 'HOUSEHOLD_WELFARE_QUINTILE',
                            'HOUSEHOLD_COMPOSITION_BY_AGE', 'HOUSEHOLD_HEAD_SEX']

        print("\n" + "-" * 60)
        print("ADDITIONAL INSIGHTS - Country overlap across different dimensions:")
        print("-" * 60)

        for dim in other_dimensions:
            if dim in df.columns:
                # Count countries per dimension
                dim_counts = df.groupby(dim)[country_col].nunique()
                if len(dim_counts) > 1 and len(dim_counts) <= 20:
                    print(f"\nCountries per '{dim}':")
                    for category in dim_counts.index[:10]:  # Show top 10
                        if pd.notna(category):
                            print(f"  {category}: {dim_counts[category]} countries")
    else:
        print("Cannot determine country overlaps without a valid country column.")

    # Data quality summary
    print("\n" + "-" * 60)
    print("DATA QUALITY SUMMARY")
    print("-" * 60)
    print(f"Missing values per column:")
    missing_data = df.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing:,} missing ({missing / len(df) * 100:.1f}%)")

    # Summary statistics
    print("\n" + "-" * 60)
    print("BASIC STATISTICS")
    print("-" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


def main():
    """Main function to run the EDA script"""

    print("=" * 60)
    print("DATA EXPLORATORY ANALYSIS TOOL")
    print("=" * 60)

    # Check if pandas is installed
    try:
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
    except ImportError:
        print("❌ Pandas is not installed. Please install it: pip install pandas")
        sys.exit(1)

    # Ask for file path
    print("\nPlease provide the path to your data file.")
    print("Supported formats: .csv, .xlsx, .xls")
    file_path = input("File path: ").strip()

    # Remove quotes if user pasted with quotes
    file_path = file_path.strip('"').strip("'")

    # Expand user directory if needed (for ~ on Mac)
    file_path = os.path.expanduser(file_path)

    # Load the data
    df = load_data(file_path)

    # Get file name for display
    file_name = os.path.basename(file_path)

    # Perform EDA
    perform_eda(df, file_name)


if __name__ == "__main__":
    main()