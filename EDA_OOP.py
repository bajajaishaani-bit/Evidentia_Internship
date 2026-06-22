#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def load_data(file_path):
    """Load data from CSV or Excel with proper handling"""

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        print("Loading CSV file...")

        try:
            # Skip first 3 rows of metadata, then load
            df = pd.read_csv(file_path, skiprows=3)

            # Convert year columns to numeric (they might be strings with quotes)
            for col in df.columns:
                try:
                    # Check if column name is a year
                    if str(col).isdigit() or (str(col).startswith('"') and str(col).strip('"').isdigit()):
                        # Convert to numeric, errors='coerce' will turn invalid to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif file_extension in ['.xlsx', '.xls']:
        print("Loading Excel file...")
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading Excel: {e}")
            sys.exit(1)

    else:
        print(f"Unsupported file type")
        sys.exit(1)


def perform_eda(df, file_name):
    """Perform EDA to answer the three questions"""

    print("\n" + "=" * 60)
    print("EDA RESULTS")
    print("=" * 60)

    print(f"\nFile analyzed: {file_name}")
    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn names (first 20): {list(df.columns)[:20]}")

    # Preview first few rows
    print("\n" + "-" * 60)
    print("PREVIEW OF FIRST 5 ROWS:")
    print("-" * 60)
    print(df.head())

    # Question 1: Which year has the most observations?
    print("\n" + "-" * 60)
    print("QUESTION 1: Which year has the most observations?")
    print("-" * 60)

    # Find year columns (columns that are years or contain year-like data)
    year_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        # Check if column name is a year (e.g., '1960', '2000', '2023')
        if col_str.isdigit() and 1960 <= int(col_str) <= 2025:
            year_cols.append(col_str)
        # Also check for columns with 'year' in name
        elif 'year' in col_str.lower():
            year_cols.append(col_str)

    if year_cols:
        print(f"Found {len(year_cols)} year columns")

        # Count non-null values per year
        year_counts = {}
        for year in year_cols:
            # Count non-null and non-zero values
            count = df[year].notna().sum()
            # Also count values that are > 0 (actual data)
            if count > 0:
                year_counts[year] = count

        if year_counts:
            print(f"\nObservations per year (showing years with data):")
            # Sort by year
            for year in sorted(year_counts.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                print(f"  {year}: {year_counts[year]:,} observations")

            # Find year with most observations
            max_year = max(year_counts, key=year_counts.get)
            max_count = year_counts[max_year]
            print(f"\n✓ Year with most observations: {max_year} ({max_count:,} observations)")
        else:
            # Try checking if data exists in year columns
            print("Checking for actual data in year columns...")
            for year in year_cols[:5]:
                sample = df[year].dropna().head(3)
                if len(sample) > 0:
                    print(f"  Sample data for {year}: {sample.tolist()}")
    else:
        print("⚠ No year columns found")

    # Question 2: How many countries are we looking at?
    print("\n" + "-" * 60)
    print("QUESTION 2: How many countries are we looking at?")
    print("-" * 60)

    # Look for country column (common names)
    country_keywords = ['country', 'nation', 'region', 'location', 'area']
    country_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in country_keywords):
            country_col = col
            break

    if country_col is None:
        # Try first column if nothing else works
        country_col = df.columns[0]
        print(f"Using first column as country: '{country_col}'")
    else:
        print(f"Using country column: '{country_col}'")

    # Clean country names and count unique
    unique_countries = df[country_col].dropna().unique()
    # Remove empty strings
    unique_countries = [c for c in unique_countries if str(c).strip() != '']

    print(f"✓ Number of unique countries/regions: {len(unique_countries)}")

    if len(unique_countries) <= 50:
        print(f"\nList of countries/regions:")
        for i, country in enumerate(sorted([str(c) for c in unique_countries]), 1):
            print(f"  {i}. {country}")
    else:
        print(f"\nFirst 20 countries/regions:")
        for i, country in enumerate(sorted([str(c) for c in unique_countries])[:20], 1):
            print(f"  {i}. {country}")
        print(f"  ... and {len(unique_countries) - 20} more")

    # Question 3: How many countries overlap?
    print("\n" + "-" * 60)
    print("QUESTION 3: How many countries overlap?")
    print("-" * 60)

    print("This is a single dataset. To check overlap between different datasets,")
    print("please run this script on multiple files and compare the country lists.")
    print(f"\nCurrent dataset has {len(unique_countries)} countries.")

    # Additional: Show data summary for recent years (2000 onwards)
    print("\n" + "-" * 60)
    print("DATA SUMMARY (2000 onwards)")
    print("-" * 60)

    recent_years = [str(year) for year in range(2000, 2026) if str(year) in df.columns]

    if recent_years:
        print(f"\nSummary for years: {', '.join(recent_years[:10])}{'...' if len(recent_years) > 10 else ''}")

        for year in recent_years[:10]:  # Show first 10 recent years
            # Convert to numeric and get stats
            values = pd.to_numeric(df[year], errors='coerce')
            values = values[values > 0]  # Only positive values
            if len(values) > 0:
                print(f"\n  Year {year}:")
                print(f"    Countries with data: {len(values)}")
                print(f"    Mean: {values.mean():.2f}")
                print(f"    Median: {values.median():.2f}")
                print(f"    Range: {values.min():.2f} - {values.max():.2f}")
            else:
                print(f"\n  Year {year}: No valid data")
    else:
        print("No recent year columns found")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


def main():
    print("=" * 60)
    print("DATA EXPLORATORY ANALYSIS TOOL")
    print("=" * 60)

    try:
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not installed")
        sys.exit(1)

    print("\nPlease provide the path to your data file.")
    print("Supported formats: .csv, .xlsx, .xls")
    file_path = input("File path: ").strip()
    file_path = file_path.strip('"').strip("'")
    file_path = os.path.expanduser(file_path)

    df = load_data(file_path)
    file_name = os.path.basename(file_path)
    perform_eda(df, file_name)


if __name__ == "__main__":
    main()