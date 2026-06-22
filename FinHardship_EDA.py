#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# ── 1. LOAD ──────────────────────────────────────────────────────────────────

def load_data(path):
    path = os.path.expanduser(path.strip().strip("'\""))
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        return pd.read_csv(path), os.path.basename(path)

    if ext in ('.xlsx', '.xls'):
        xf = pd.ExcelFile(path, engine='openpyxl')
        frames = []
        for sheet in xf.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
            if not df.empty:
                df['source_sheet'] = sheet
                frames.append(df)
        if not frames:
            sys.exit("No data found in any sheet.")
        return pd.concat(frames, ignore_index=True), os.path.basename(path)

    sys.exit("Unsupported format. Use .csv, .xlsx, or .xls")


# ── 2. HELPER: find a column matching keywords ────────────────────────────────

def find_col(df, keywords):
    """Return first column whose name contains any keyword (case-insensitive)."""
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None


# ── 3. EDA ────────────────────────────────────────────────────────────────────

def eda(df, name):
    sep = lambda title: print(f"\n{'─'*55}\n{title}\n{'─'*55}")

    # Basic shape
    sep("OVERVIEW")
    print(f"File  : {name}")
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Cols  : {list(df.columns)}")
    print(df.head(3).to_string())

    # ── Q1: Year with most observations ──────────────────────────────────────
    sep("Q1 · Year with most observations")
    year_col = find_col(df, ['year', 'yr', 'date', 'time', 'period'])

    if year_col:
        years = pd.to_numeric(df[year_col], errors='coerce')
        if years.isna().all():                      # maybe it's a date string
            years = pd.to_datetime(df[year_col], errors='coerce').dt.year
        counts = years.value_counts().sort_index().dropna()
        print(counts.rename("observations").to_string())
        best = int(counts.idxmax())
        print(f"\n→ Most observations: {best} ({int(counts.max()):,} rows)")
    else:
        print("No year/date column found.")
        counts, year_col = None, None

    # ── Q2: How many countries ────────────────────────────────────────────────
    sep("Q2 · Unique countries / regions")
    country_col = find_col(df, ['country', 'nation', 'region', 'location',
                                 'area', 'state', 'spatialdimension'])
    if country_col:
        unique = sorted(df[country_col].dropna().astype(str).unique())
        print(f"Column used : '{country_col}'")
        print(f"Count       : {len(unique)}")
        for i, c in enumerate(unique, 1):
            print(f"  {i:>3}. {c}")
    else:
        print("No country column found.")
        unique = []

    # ── Q3: Country overlap across years ─────────────────────────────────────
    sep("Q3 · Country overlap across years")
    if country_col and year_col:
        clean = df[[country_col, year_col]].copy()
        clean[year_col] = pd.to_numeric(clean[year_col], errors='coerce')
        clean = clean.dropna()
        all_years = clean[year_col].unique()

        per_year = clean.groupby(year_col)[country_col].nunique()
        print("Countries per year:")
        print(per_year.to_string())

        if len(all_years) > 1:
            sets = [set(clean[clean[year_col] == y][country_col].astype(str))
                    for y in all_years]
            in_all = set.intersection(*sets)
            print(f"\nIn ALL {len(all_years)} years : {len(in_all)} countries")

            # In at least 2 years
            multi = (clean.groupby(country_col)[year_col]
                         .nunique()
                         .pipe(lambda s: s[s >= 2]))
            print(f"In ≥2 years           : {len(multi)} countries")
    else:
        print("Need both year and country columns to check overlap.")

    # ── Data quality ──────────────────────────────────────────────────────────
    sep("DATA QUALITY · Missing values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values.")
    else:
        pct = (missing / len(df) * 100).round(1)
        print(pd.DataFrame({'missing': missing, '%': pct}).to_string())

    # ── Basic stats ───────────────────────────────────────────────────────────
    sep("NUMERIC SUMMARY")
    print(df.select_dtypes(include=np.number).describe().round(2).to_string())

    # ── VISUALISATIONS ────────────────────────────────────────────────────────
    sep("VISUALISATIONS")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"EDA — {name}", fontsize=13, fontweight='bold')

    # Plot 1: Observations per year (bar chart)
    ax = axes[0]
    if year_col and counts is not None and len(counts):
        counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
        ax.set_title("Observations per year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, "No year data", ha='center', va='center')
        ax.set_title("Observations per year")

    # Plot 2: Distribution of the main numeric variable (OBS_VALUE or first numeric col)
    ax = axes[1]
    num_col = find_col(df, ['obs_value', 'value', 'obs']) or (
        df.select_dtypes(include=np.number).columns.tolist()[0]
        if not df.select_dtypes(include=np.number).empty else None
    )
    if num_col:
        df[num_col].dropna().plot(kind='hist', bins=30, ax=ax,
                                   color='mediumseagreen', edgecolor='white')
        ax.set_title(f"Distribution of {num_col}")
        ax.set_xlabel(num_col)
        ax.set_ylabel("Frequency")
        # Mark median
        med = df[num_col].median()
        ax.axvline(med, color='darkred', linestyle='--', linewidth=1.2,
                   label=f"Median: {med:.1f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No numeric column", ha='center', va='center')
        ax.set_title("Value distribution")

    # Plot 3: Countries per year (if available)
    ax = axes[2]
    if country_col and year_col:
        per_year.plot(kind='line', marker='o', ax=ax, color='darkorange')
        ax.set_title("Countries per year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Unique countries")
    else:
        ax.text(0.5, 0.5, "No year/country data", ha='center', va='center')
        ax.set_title("Countries per year")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved → {out_path}")


# ── 4. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("EDA TOOL\nSupported formats: .csv  .xlsx  .xls\n")
    path = input("File path: ")
    df, name = load_data(path)
    eda(df, name)