"""
src/eda.py
----------
Exploratory Data Analysis module.

Every function returns a clean DataFrame or Series so results can be
chained, saved, or passed directly into visualization functions.
"""

import pandas as pd
import numpy as np


# ── 1. Overview ───────────────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """High-level KPIs — first thing a business stakeholder wants to see."""
    stats = {
        "Total Revenue ($)":    f"${df['revenue'].sum():,.0f}",
        "Total Profit ($)":     f"${df['profit'].sum():,.0f}",
        "Total Orders":         f"{len(df):,}",
        "Unique Customers":     f"{df['customer_id'].nunique():,}",
        "Avg Order Value ($)":  f"${df['revenue'].mean():,.2f}",
        "Avg Profit Margin":    f"{df['profit_margin'].mean():.1%}",
        "Top Category":         df.groupby('category')['revenue'].sum().idxmax(),
        "Top Region":           df.groupby('region')['revenue'].sum().idxmax(),
    }
    result = pd.DataFrame(stats.items(), columns=["Metric", "Value"])
    print(result.to_string(index=False))
    return result


# ── 2. Sales trend over time ──────────────────────────────────────────────────

def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate revenue and profit by year-month.
    WHY: Monthly granularity captures seasonality without day-level noise.
    """
    trend = (
        df.groupby("year_month", sort=True)
          .agg(revenue=("revenue", "sum"),
               profit=("profit",  "sum"),
               orders=("order_id", "count"))
          .reset_index()
    )
    trend["year_month_dt"] = pd.to_datetime(trend["year_month"])
    trend.sort_values("year_month_dt", inplace=True)
    return trend


def yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year revenue comparison."""
    return (
        df.groupby("year")["revenue"]
          .sum()
          .reset_index()
          .rename(columns={"revenue": "total_revenue"})
    )


# ── 3. Product analysis ───────────────────────────────────────────────────────

def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top N products by revenue with order count and margin.
    WHY: Revenue alone can be misleading — a product with many orders
         but thin margin needs a different strategy than a high-ticket item.
    """
    return (
        df.groupby("product")
          .agg(
              revenue=("revenue",       "sum"),
              profit=("profit",         "sum"),
              orders=("order_id",       "count"),
              avg_margin=("profit_margin", "mean"),
          )
          .sort_values("revenue", ascending=False)
          .head(n)
          .reset_index()
    )


def bottom_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Underperformers by profit — actionable for discontinuation."""
    return (
        df.groupby("product")
          .agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
          .sort_values("profit")
          .head(n)
          .reset_index()
    )


# ── 4. Category analysis ──────────────────────────────────────────────────────

def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue share + profitability by category."""
    cat = (
        df.groupby("category")
          .agg(
              revenue=("revenue",       "sum"),
              profit=("profit",         "sum"),
              orders=("order_id",       "count"),
              avg_margin=("profit_margin", "mean"),
          )
          .reset_index()
    )
    cat["revenue_share"] = (cat["revenue"] / cat["revenue"].sum()).round(4)
    return cat.sort_values("revenue", ascending=False)


# ── 5. Region analysis ────────────────────────────────────────────────────────

def region_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order volume by region."""
    return (
        df.groupby("region")
          .agg(
              revenue=("revenue", "sum"),
              profit=("profit",  "sum"),
              orders=("order_id", "count"),
              customers=("customer_id", "nunique"),
              avg_order_value=("revenue", "mean"),
          )
          .sort_values("revenue", ascending=False)
          .reset_index()
    )


def city_performance(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N cities by revenue."""
    return (
        df.groupby(["region", "city"])["revenue"]
          .sum()
          .reset_index()
          .sort_values("revenue", ascending=False)
          .head(n)
    )


# ── 6. Customer behavior ──────────────────────────────────────────────────────

def top_customers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top N customers by revenue — classic RFM precursor.
    Highlights who the high-value accounts are.
    """
    return (
        df.groupby("customer_id")
          .agg(
              total_revenue=("revenue",  "sum"),
              total_orders=("order_id", "count"),
              avg_order_value=("revenue", "mean"),
              total_profit=("profit",   "sum"),
          )
          .sort_values("total_revenue", ascending=False)
          .head(n)
          .reset_index()
    )


def segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order breakdown by customer segment."""
    seg = (
        df.groupby("segment")
          .agg(
              revenue=("revenue", "sum"),
              orders=("order_id", "count"),
              customers=("customer_id", "nunique"),
          )
          .reset_index()
    )
    seg["revenue_per_customer"] = (seg["revenue"] / seg["customers"]).round(2)
    return seg.sort_values("revenue", ascending=False)


# ── 7. Seasonal analysis ──────────────────────────────────────────────────────

def monthly_avg_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average revenue per month (across all years).
    WHY: Averaging removes year-to-year growth bias,
         isolating the pure seasonal pattern.
    """
    MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = (
        df.groupby(["year", "month", "month_name"])["revenue"]
          .sum()
          .groupby(level=["month", "month_name"])
          .mean()
          .reset_index()
          .rename(columns={"revenue": "avg_revenue"})
    )
    monthly["month_name"] = pd.Categorical(monthly["month_name"],
                                           categories=MONTH_ORDER, ordered=True)
    return monthly.sort_values("month_name")


def heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: rows = month, columns = region, values = revenue.
    Used for the month × region heatmap.
    """
    MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = (
        df.groupby(["month_name", "region"])["revenue"]
          .sum()
          .unstack(fill_value=0)
    )
    pivot = pivot.reindex([m for m in MONTH_ORDER if m in pivot.index])
    return pivot


# ── 8. Order distribution ─────────────────────────────────────────────────────

def order_value_distribution(df: pd.DataFrame) -> pd.Series:
    """Returns the raw revenue series for histogram plotting."""
    return df["revenue"]


def discount_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare avg profit margin across discount tiers.
    WHY: Shows whether heavy discounting is actually hurting profitability.
    """
    bins   = [-0.01, 0, 0.10, 0.20, 0.30]
    labels = ["No Discount", "5–10%", "11–20%", "21–30%"]
    df = df.copy()
    df["discount_tier"] = pd.cut(df["discount"], bins=bins, labels=labels)
    return (
        df.groupby("discount_tier", observed=True)
          .agg(
              avg_margin=("profit_margin", "mean"),
              avg_revenue=("revenue",       "mean"),
              orders=("order_id",           "count"),
          )
          .reset_index()
    )


# ── 9. Insights report ────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> list[str]:
    """
    Derive text insights programmatically.
    Returns a list of strings — written to outputs/report.txt.
    """
    insights = []

    # Peak month
    monthly = monthly_avg_revenue(df)
    peak    = monthly.loc[monthly["avg_revenue"].idxmax(), "month_name"]
    insights.append(
        f"INSIGHT 1 – Peak Month: '{peak}' records the highest average revenue. "
        f"Recommendation: Scale up inventory, marketing spend, and support staff "
        f"in the weeks leading up to {peak}."
    )

    # Top region
    reg     = region_performance(df)
    top_reg = reg.iloc[0]
    low_reg = reg.iloc[-1]
    insights.append(
        f"INSIGHT 2 – Region Performance: '{top_reg['region']}' leads with "
        f"${top_reg['revenue']:,.0f} in revenue. "
        f"'{low_reg['region']}' is the weakest at ${low_reg['revenue']:,.0f}. "
        f"Recommendation: Investigate low-performing region's logistics and lead generation."
    )

    # Category margins
    cat = category_performance(df)
    best_margin_cat = cat.sort_values("avg_margin", ascending=False).iloc[0]
    insights.append(
        f"INSIGHT 3 – Best Margin Category: '{best_margin_cat['category']}' has "
        f"the highest avg profit margin ({best_margin_cat['avg_margin']:.1%}). "
        f"Recommendation: Prioritise this category in upselling and cross-selling campaigns."
    )

    # Top product
    tp = top_products(df, 1).iloc[0]
    insights.append(
        f"INSIGHT 4 – Top Product: '{tp['product']}' generates ${tp['revenue']:,.0f} revenue "
        f"across {tp['orders']:,} orders. "
        f"Recommendation: Ensure consistent stock availability and consider bundling."
    )

    # Discount impact
    disc = discount_impact(df)
    no_disc = disc[disc["discount_tier"] == "No Discount"]["avg_margin"].values
    high_disc = disc[disc["discount_tier"] == "21–30%"]["avg_margin"].values
    if len(no_disc) and len(high_disc):
        diff = no_disc[0] - high_disc[0]
        insights.append(
            f"INSIGHT 5 – Discount Impact: Orders with 21–30% discount have "
            f"{diff:.1%} lower profit margins than non-discounted orders. "
            f"Recommendation: Cap discounts at 10% unless clearing overstock."
        )

    # Customer concentration
    cust    = top_customers(df, 10)
    top10_rev = cust["total_revenue"].sum()
    total_rev = df["revenue"].sum()
    pct = top10_rev / total_rev
    insights.append(
        f"INSIGHT 6 – Customer Concentration: Top 10 customers account for "
        f"{pct:.1%} of total revenue (${top10_rev:,.0f}). "
        f"Recommendation: Build a VIP retention programme for these accounts."
    )

    # YoY growth
    yoy = yoy_growth(df).sort_values("year")
    if len(yoy) >= 2:
        growth = (yoy.iloc[-1]["total_revenue"] - yoy.iloc[-2]["total_revenue"]) \
                 / yoy.iloc[-2]["total_revenue"]
        insights.append(
            f"INSIGHT 7 – YoY Growth: Revenue grew {growth:.1%} from "
            f"{int(yoy.iloc[-2]['year'])} to {int(yoy.iloc[-1]['year'])}. "
            f"Recommendation: {'Accelerate growth with paid acquisition.' if growth > 0 else 'Audit customer churn and pricing strategy.'}"
        )

    return insights
