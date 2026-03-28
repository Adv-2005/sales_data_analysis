"""
src/data_cleaning.py
--------------------
Handles all data ingestion and cleaning steps.

Design: Every function is pure (input → output), making it easy to
test, reuse, and explain in interviews.
"""

import pandas as pd
import numpy as np


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame and do a quick sanity check.
    Returns raw DataFrame — no transformations yet.
    """
    df = pd.read_csv(filepath)
    print(f"[load]  Shape: {df.shape}  |  Columns: {list(df.columns)}")
    return df


# ── 2. Standardise column names ───────────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace, lowercase, replace spaces with underscores.
    WHY: Prevents KeyError surprises from 'Revenue ' vs 'revenue'.
    """
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    print(f"[cols]  Standardised → {list(df.columns)}")
    return df


# ── 3. Fix data types ─────────────────────────────────────────────────────────

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to correct types.
    WHY: Pandas reads everything as object by default; wrong types
         break aggregations and date math.
    """
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["revenue"]    = pd.to_numeric(df["revenue"],    errors="coerce")
    df["profit"]     = pd.to_numeric(df["profit"],     errors="coerce")
    df["quantity"]   = pd.to_numeric(df["quantity"],   errors="coerce").astype("Int64")
    df["discount"]   = pd.to_numeric(df["discount"],   errors="coerce")
    print(f"[types] Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── 4. Handle missing values ──────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Targeted strategy per column — never blindly drop rows.

    Strategy:
    - discount   → 0  (no discount applied; safest business assumption)
    - city       → 'Unknown'  (region is still intact for aggregation)
    - segment    → mode  (most common segment fills the gap)
    - numeric NaN from coercion → drop (data is corrupted at source)
    """
    before = len(df)

    df["discount"] = df["discount"].fillna(0.0)
    df["city"]     = df["city"].fillna("Unknown")
    df["segment"]  = df["segment"].fillna(df["segment"].mode()[0])

    # Drop rows where core numeric fields couldn't be parsed
    df.dropna(subset=["revenue", "unit_price", "quantity", "date"], inplace=True)

    print(f"[null]  Rows before: {before:,}  |  After: {len(df):,}  |  Dropped: {before - len(df):,}")
    return df


# ── 5. Remove duplicates ──────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Duplicate rows inflate revenue figures in aggregations.
    Keep first occurrence to preserve chronological order.
    """
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[dupes] Removed: {before - len(df):,}  |  Remaining: {len(df):,}")
    return df


# ── 6. Fix data quality issues ────────────────────────────────────────────────

def fix_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business-rule validation:
    - Negative quantity → invalid order, remove
    - category column may have been lowercased; restore Title Case
    - Recompute revenue from unit_price × quantity × (1 - discount)
      so downstream analysis uses a consistent source of truth
    """
    # Remove invalid quantities
    before = len(df)
    df = df[df["quantity"] > 0].copy()
    print(f"[quality] Removed {before - len(df):,} rows with quantity ≤ 0")

    # Normalise category casing
    df["category"] = df["category"].str.title()

    # Recompute revenue (guards against import inconsistencies)
    df["revenue"] = (df["unit_price"] * df["quantity"] * (1 - df["discount"])).round(2)

    return df


# ── 7. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and business features.
    WHY: Date decomposition enables trend analysis without extra groupby magic.
    """
    df["year"]          = df["date"].dt.year
    df["month"]         = df["date"].dt.month
    df["month_name"]    = df["date"].dt.strftime("%b")
    df["quarter"]       = df["date"].dt.quarter
    df["year_month"]    = df["date"].dt.to_period("M").astype(str)
    df["profit_margin"] = (df["profit"] / df["revenue"].replace(0, np.nan)).round(4)

    print(f"[feat]  Added: year, month, quarter, year_month, profit_margin")
    return df


# ── 8. Master pipeline ────────────────────────────────────────────────────────

def run_cleaning_pipeline(filepath: str) -> pd.DataFrame:
    """
    Chains all steps in order. Call this from notebooks/scripts.

    Returns: clean, feature-enriched DataFrame
    """
    df = load_data(filepath)
    df = standardize_columns(df)
    df = fix_dtypes(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = fix_quality_issues(df)
    df = engineer_features(df)

    print(f"\n✓ Cleaning complete. Final shape: {df.shape}")
    return df


# ── Diagnostic helper ─────────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> None:
    """Print a concise data quality summary — useful for README screenshots."""
    print("\n── Data Quality Report ─────────────────────────────")
    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Null counts :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Duplicates  : {df.duplicated().sum()}")
    print(f"  Revenue range: ${df['revenue'].min():,.2f} → ${df['revenue'].max():,.2f}")
    print("────────────────────────────────────────────────────\n")
