"""
run_analysis.py
---------------
Master script — run this to execute the full pipeline end-to-end.

Usage:
    python run_analysis.py

Produces:
    outputs/charts/*.png      — 8 publication-quality charts
    outputs/report.txt        — Business insights report
"""

import sys
import os
import time

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import src.data_cleaning  as cleaning
import src.eda             as eda
import src.visualization   as viz

REPORT_PATH = "outputs/report.txt"


# ── Report Writer ─────────────────────────────────────────────────────────────

def write_report(insights: list[str], summary_df, df) -> None:
    """
    Persist insights to a text file.
    WHY: A written artifact demonstrates communication skills — not just code.
    """
    os.makedirs("outputs", exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("   SALES DATA ANALYSIS — BUSINESS INSIGHTS REPORT\n")
        f.write("=" * 65 + "\n\n")

        f.write("── EXECUTIVE SUMMARY ─────────────────────────────────────\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"  {row['Metric']:<30} {row['Value']}\n")

        f.write("\n\n── KEY INSIGHTS & RECOMMENDATIONS ────────────────────────\n\n")
        for insight in insights:
            f.write(f"  {'─' * 60}\n")
            f.write(f"  {insight}\n\n")

        f.write("\n── DISCOUNT IMPACT ANALYSIS ──────────────────────────────\n\n")
        disc = eda.discount_impact(df)
        f.write(disc.to_string(index=False))
        f.write("\n")

        f.write("\n\n── FUTURE ENHANCEMENTS (INTERVIEW TALKING POINTS) ────────\n")
        f.write("  1. Streamlit dashboard for real-time interactive exploration\n")
        f.write("  2. Prophet / ARIMA time-series forecasting for demand planning\n")
        f.write("  3. ML model (XGBoost) for customer churn prediction\n")
        f.write("  4. RFM segmentation (Recency, Frequency, Monetary) for CRM\n")
        f.write("  5. Automated alerting when regional revenue dips > 15% MoM\n")
        f.write("\n" + "=" * 65 + "\n")

    print(f"  ✓ Report saved → {REPORT_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("\n" + "=" * 55)
    print("  SALES ANALYSIS PIPELINE — START")
    print("=" * 55)

    # Step 1 — Clean
    print("\n[STEP 1] Data Cleaning")
    print("-" * 40)
    df = cleaning.run_cleaning_pipeline("data/sales_raw.csv")
    cleaning.data_quality_report(df)

    # Save clean dataset
    clean_path = "data/sales_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"  ✓ Clean dataset saved → {clean_path}")

    # Step 2 — EDA
    print("\n[STEP 2] Exploratory Data Analysis")
    print("-" * 40)
    summary = eda.summary_stats(df)

    print("\n  — Top 5 Products —")
    print(eda.top_products(df, 5)[["product","revenue","profit","avg_margin"]].to_string(index=False))

    print("\n  — Region Performance —")
    print(eda.region_performance(df)[["region","revenue","profit","orders"]].to_string(index=False))

    print("\n  — Category Performance —")
    print(eda.category_performance(df)[["category","revenue","revenue_share","avg_margin"]].to_string(index=False))

    print("\n  — Segment Analysis —")
    print(eda.segment_analysis(df).to_string(index=False))

    # Step 3 — Visualisations
    print("\n[STEP 3] Visualisations")
    print("-" * 40)
    viz.generate_all_charts(df, eda)

    # Step 4 — Insights Report
    print("\n[STEP 4] Insights Report")
    print("-" * 40)
    insights = eda.generate_insights(df)
    for i, ins in enumerate(insights, 1):
        print(f"\n  {ins}")
    write_report(insights, summary, df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 55}")
    print(f"  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
