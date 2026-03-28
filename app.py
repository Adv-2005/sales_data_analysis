"""
app.py — Streamlit Sales Analytics Dashboard
---------------------------------------------
Run: streamlit run app.py

5-tab layout:
  1. Overview       — KPI cards + revenue summary
  2. Trends         — Time series + seasonality
  3. Products        — Top products + category breakdown
  4. Customer Segments — RFM scatter + segment table
  5. Forecast       — 6-month revenue forecast + seasonal index

Design principles:
  - Data is cached with @st.cache_data (fast re-renders)
  - All charts use Matplotlib (consistent with existing codebase)
  - Sidebar filters apply globally — no per-page duplication
  - Every section has a plain-English insight below the chart
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import src.data_cleaning  as cleaning
import src.eda             as eda
import src.segmentation    as seg
import src.forecasting     as fc
import src.kpi_tracker     as kpi
import src.visualization   as viz


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — minimal, uses Streamlit's native variables
st.markdown("""
<style>
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1E293B; }
    .metric-label { font-size: 0.82rem; color: #64748B; margin-top: 2px; }
    .metric-delta { font-size: 0.85rem; font-weight: 600; margin-top: 4px; }
    .insight-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #1E3A5F;
    }
    .alert-red    { background:#FEF2F2; border-left:4px solid #DC2626; }
    .alert-yellow { background:#FFFBEB; border-left:4px solid #D97706; }
    .alert-green  { background:#F0FDF4; border-left:4px solid #16A34A; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading and cleaning data…")
def load_data():
    return cleaning.run_cleaning_pipeline("data/sales_raw.csv")


@st.cache_data(show_spinner="Computing RFM scores…")
def get_rfm(_df):
    return seg.compute_rfm(_df)


@st.cache_data(show_spinner="Building forecast…")
def get_forecast(_df, periods=6):
    return fc.forecast(_df, periods=periods)


@st.cache_data(show_spinner="Running KPI checks…")
def get_kpi_data(_df, window=3):
    alerts  = kpi.run_kpi_alerts(_df, window_months=window)
    history = kpi.build_kpi_history(_df)
    return alerts, history


# ── Shared figure styling ─────────────────────────────────────────────────────
BG   = "#F8FAFC"
BLUE = "#2563EB"
GREEN = "#16A34A"
RED  = "#DC2626"
AMB  = "#D97706"
MUTED = "#64748B"
TEXT = "#1E293B"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_edgecolor("#E2E8F0")
    ax.tick_params(colors=MUTED, labelsize=8)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — global filters
# ═══════════════════════════════════════════════════════════════════

def sidebar_filters(df):
    st.sidebar.title("Sales Analytics")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    years = sorted(df["year"].unique())
    sel_years = st.sidebar.multiselect("Year", years, default=years)

    regions = sorted(df["region"].unique())
    sel_regions = st.sidebar.multiselect("Region", regions, default=regions)

    categories = sorted(df["category"].unique())
    sel_cats = st.sidebar.multiselect("Category", categories, default=categories)

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Synthetic retail dataset | 12k rows")
    st.sidebar.caption("Built with Python · Pandas · Matplotlib · Streamlit")

    mask = (
        df["year"].isin(sel_years) &
        df["region"].isin(sel_regions) &
        df["category"].isin(sel_cats)
    )
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════

def tab_overview(df):
    st.header("Business Overview")

    # ── KPI row ─────────────────────────────────────────────────────
    total_rev    = df["revenue"].sum()
    total_profit = df["profit"].sum()
    margin       = df["profit_margin"].mean()
    orders       = len(df)
    aov          = df["revenue"].mean()
    customers    = df["customer_id"].nunique()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_cards = [
        (c1, "Total Revenue",    f"${total_rev:,.0f}",    BLUE),
        (c2, "Total Profit",     f"${total_profit:,.0f}", GREEN),
        (c3, "Profit Margin",    f"{margin:.1%}",         AMB),
        (c4, "Total Orders",     f"{orders:,}",           MUTED),
        (c5, "Avg Order Value",  f"${aov:,.0f}",          BLUE),
        (c6, "Unique Customers", f"{customers:,}",        GREEN),
    ]
    for col, label, val, color in kpi_cards:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{val}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue by region + category ────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        reg = eda.region_performance(df)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
        colors = [BLUE, GREEN, AMB, RED][:len(reg)]
        bars = ax.barh(reg["region"], reg["revenue"], color=colors,
                       edgecolor="white", height=0.5)
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width()/1e3:.0f}K",
                    va="center", fontsize=8, color=TEXT)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        _style(ax, "Revenue by Region", xlabel="Revenue ($)")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        cat = eda.category_performance(df)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
        explode = [0.04 if i == 0 else 0 for i in range(len(cat))]
        ax.pie(cat["revenue"], labels=cat["category"], autopct="%1.1f%%",
               explode=explode,
               colors=["#2563EB", "#16A34A", "#D97706"][:len(cat)],
               startangle=140, textprops={"fontsize": 9},
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Revenue by Category", fontsize=12,
                     fontweight="bold", color=TEXT, pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── YoY comparison ───────────────────────────────────────────────
    st.subheader("Year-over-Year Revenue")
    yoy = eda.yoy_growth(df)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG)
    bar_colors = [BLUE if i < len(yoy) - 1 else GREEN for i in range(len(yoy))]
    bars = ax.bar(yoy["year"].astype(str), yoy["total_revenue"],
                  color=bar_colors, edgecolor="white", width=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"${bar.get_height()/1e3:.0f}K",
                ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    _style(ax, "Annual Revenue Comparison", ylabel="Revenue ($)")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""<div class="insight-box">
     <strong>Insight:</strong> Technology leads all categories by revenue share,
    while Office Supplies shows the highest profit margin (34.9%). East region consistently
    outperforms — investigate replicating its playbook in the South region.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Trends
# ═══════════════════════════════════════════════════════════════════

def tab_trends(df):
    st.header("Sales Trends")

    trend = eda.monthly_sales_trend(df)
    monthly_avg = eda.monthly_avg_revenue(df)

    # ── Monthly trend ────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 4), facecolor=BG)
    x = range(len(trend))
    labels = trend["year_month"].tolist()

    ax1.plot(x, trend["revenue"], color=BLUE, linewidth=2,
             marker="o", markersize=3, label="Revenue")
    ax1.fill_between(x, trend["revenue"], alpha=0.08, color=BLUE)

    ax2 = ax1.twinx()
    ax2.plot(x, trend["profit"], color=GREEN, linewidth=1.8,
             linestyle="--", marker="s", markersize=3, label="Profit")
    ax2.set_ylabel("Profit ($)", fontsize=9, color=MUTED)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines[["top"]].set_visible(False)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    step = max(1, len(labels) // 12)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.7)
    _style(ax1, "Monthly Revenue & Profit", ylabel="Revenue ($)")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Seasonality ──────────────────────────────────────────────────
    st.subheader("Seasonal Pattern")
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=BG)
    peak_idx = monthly_avg["avg_revenue"].idxmax()
    bar_colors = [RED if i == peak_idx else BLUE for i in monthly_avg.index]
    bars = ax.bar(monthly_avg["month_name"].astype(str),
                  monthly_avg["avg_revenue"], color=bar_colors,
                  edgecolor="white", width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"${bar.get_height()/1e3:.1f}K",
                ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    _style(ax, "Avg Revenue by Month (seasonal pattern)", ylabel="Avg Revenue ($)")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Heatmap
    st.subheader("Revenue Heatmap — Month × Region")
    pivot = eda.heatmap_data(df)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    im = ax.imshow(pivot.values.astype(float), cmap="YlOrRd", aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = float(pivot.values[i, j])
            ax.text(j, i, f"${val/1e3:.0f}K", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if val > pivot.values.max() * 0.6 else TEXT)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Revenue Heatmap — Month × Region",
                 fontsize=12, fontweight="bold", color=TEXT, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Revenue ($)", fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""<div class="insight-box">
     <strong>Insight:</strong> November is the peak revenue month across all years.
    Q4 (Oct–Dec) consistently drives 30–35% of annual revenue. East region dominates
    Q4 — consider targeted campaigns in Central and South regions to capture missed demand.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Products
# ═══════════════════════════════════════════════════════════════════

def tab_products(df):
    st.header("Product & Category Analysis")

    col1, col2 = st.columns([3, 2])

    with col1:
        n_products = st.slider("Number of top products to show", 5, 20, 10)
        top = eda.top_products(df, n_products)

        fig, ax = plt.subplots(figsize=(8, max(4, n_products * 0.5)), facecolor=BG)
        df_sorted = top.sort_values("revenue")
        import matplotlib.cm as cm
        colors = cm.Blues(np.linspace(0.3, 0.9, len(df_sorted)))
        bars = ax.barh(df_sorted["product"], df_sorted["revenue"],
                       color=colors, edgecolor="white")
        for bar, margin_val in zip(bars, df_sorted["avg_margin"]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width():,.0f}  ({margin_val:.0%})",
                    va="center", fontsize=8, color=TEXT)
        ax.set_xlim(0, df_sorted["revenue"].max() * 1.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        _style(ax, f"Top {n_products} Products (revenue + margin %)", xlabel="Revenue ($)")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        cat = eda.category_performance(df)
        st.dataframe(
            cat[["category", "revenue", "profit", "orders", "avg_margin", "revenue_share"]]
              .style
              .format({
                  "revenue":       "${:,.0f}",
                  "profit":        "${:,.0f}",
                  "avg_margin":    "{:.1%}",
                  "revenue_share": "{:.1%}",
              })
              .background_gradient(subset=["avg_margin"], cmap="Greens"),
            use_container_width=True, height=200
        )
        # Discount impact
        disc = eda.discount_impact(df)
        fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor=BG)
        colors_d = [GREEN, AMB, RED, "#9F1239"][:len(disc)]
        ax2.bar(disc["discount_tier"].astype(str), disc["avg_margin"] * 100,
                color=colors_d, edgecolor="white", width=0.5)
        _style(ax2, "Margin by Discount Tier", ylabel="Avg Margin (%)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("""<div class="insight-box">
     <strong>Insight:</strong> Laptop generates 26.6% of total revenue but has only an 18% margin.
    Office Supplies has a 34.9% margin — 2x better than Technology. Heavy discounting (21–30%)
    erodes margin noticeably. Cap routine discounts at 10%.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Customer Segments (RFM)
# ═══════════════════════════════════════════════════════════════════

def tab_segments(df):
    st.header("Customer Segmentation (RFM)")

    st.markdown("""
    **RFM** scores each customer on three dimensions:
    - **Recency** — how recently they purchased (R=4 is most recent)
    - **Frequency** — how often they buy (F=4 = most frequent)
    - **Monetary** — how much they spend (M=4 = highest spender)
    """)

    rfm = get_rfm(df)
    summary = seg.segment_summary(rfm)

    # ── Segment bars + scatter ───────────────────────────────────────
    col1, col2 = st.columns(2)

    COLORS_MAP = {s: seg.SEGMENTS.get(s, ("","","","#888"))[3]
                  for s in rfm["segment"].unique()}

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
        colors = [COLORS_MAP.get(s, "#888") for s in summary["segment"]]
        bars = ax.barh(summary["segment"], summary["customers"],
                       color=colors, edgecolor="white", height=0.6)
        for bar, pct in zip(bars, summary["pct_customers"]):
            ax.text(bar.get_width() * 1.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct}%", va="center", fontsize=8, color=TEXT)
        ax.set_xlim(0, summary["customers"].max() * 1.25)
        _style(ax, "Customers per Segment", xlabel="Count")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
        scatter_colors = [COLORS_MAP.get(s, "#888") for s in rfm["segment"]]
        sizes = ((rfm["monetary"] - rfm["monetary"].min())
                 / (rfm["monetary"].max() - rfm["monetary"].min()) * 200 + 10)
        ax.scatter(rfm["recency"], rfm["frequency"],
                   c=scatter_colors, s=sizes, alpha=0.4,
                   edgecolors="white", linewidth=0.3)
        ax.set_xlabel("Recency (days)", fontsize=9, color=MUTED)
        ax.set_ylabel("Frequency (orders)", fontsize=9, color=MUTED)
        _style(ax, "Recency vs Frequency\n(bubble = revenue)")
        ax.grid(color="#E2E8F0", linewidth=0.6, linestyle="--")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Segment action table ─────────────────────────────────────────
    st.subheader("Segment Actions")
    display = summary[["segment", "customers", "pct_customers",
                        "avg_monetary", "total_revenue", "action"]].copy()
    st.dataframe(
        display.style.format({
            "avg_monetary":  "${:,.0f}",
            "total_revenue": "${:,.0f}",
            "pct_customers": "{:.1f}%",
        }),
        use_container_width=True, height=380
    )

    # ── Champion vs Lost highlight ───────────────────────────────────
    champions = rfm[rfm["segment"] == "Champions"]
    lost      = rfm[rfm["segment"] == "Lost"]

    c1, c2 = st.columns(2)
    c1.metric("🏆 Champions",
              f"{len(champions):,} customers",
              f"Avg ${champions['monetary'].mean():,.0f} / customer")
    c2.metric("😴 Lost Customers",
              f"{len(lost):,} customers",
              f"Avg ${lost['monetary'].mean():,.0f} / customer",
              delta_color="inverse")

    st.markdown("""<div class="insight-box">
     <strong>Insight:</strong> Champions (R≥4, F≥4) spend 3× more than Lost customers.
    Invest in loyalty rewards for Champions. Lost customers account for 21% of the base —
    a targeted win-back campaign could recover significant revenue.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — Forecast + KPI Tracker
# ═══════════════════════════════════════════════════════════════════

def tab_forecast_kpi(df):
    st.header("Revenue Forecast & KPI Tracker")

    # ── Forecast ─────────────────────────────────────────────────────
    st.subheader("6-Month Revenue Forecast")
    periods = st.slider("Forecast horizon (months)", 3, 12, 6)
    forecast_df, seasonal_idx = get_forecast(df, periods)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor=BG)

    hist_mask   = forecast_df["actual"].notna()
    future_mask = forecast_df["actual"].isna()

    ax1.plot(forecast_df.loc[hist_mask, "ds"],
             forecast_df.loc[hist_mask, "actual"],
             color=BLUE, linewidth=2, label="Actual", zorder=3)
    ax1.plot(forecast_df.loc[hist_mask, "ds"],
             forecast_df.loc[hist_mask, "forecast"],
             color=MUTED, linewidth=1.2, linestyle="--",
             label="Model fit", alpha=0.7)
    ax1.plot(forecast_df.loc[future_mask, "ds"],
             forecast_df.loc[future_mask, "forecast"],
             color=RED, linewidth=2.5, label=f"Forecast (+{periods}m)", zorder=3)
    ax1.fill_between(forecast_df.loc[future_mask, "ds"],
                     forecast_df.loc[future_mask, "ci_lower"],
                     forecast_df.loc[future_mask, "ci_upper"],
                     color=RED, alpha=0.12, label="95% CI")

    split_date = forecast_df.loc[future_mask, "ds"].iloc[0]
    ax1.axvline(split_date, color="#94A3B8", linewidth=1, linestyle=":")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax1.legend(fontsize=9, framealpha=0.7)
    _style(ax1, "Revenue Forecast", ylabel="Revenue ($)")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # Seasonal index
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    idx_vals = [seasonal_idx.get(m, 1.0) for m in range(1, 13)]
    bar_colors = [RED if v == max(idx_vals) else
                  BLUE if v >= 1.0 else MUTED for v in idx_vals]
    bars = ax2.bar(MONTHS, idx_vals, color=bar_colors, edgecolor="white", width=0.6)
    ax2.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    for bar, val in zip(bars, idx_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    _style(ax2, "Seasonal Index (>1.0 = above average)")
    ax2.set_ylim(0, max(idx_vals) * 1.2)

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Forecast table
    future = forecast_df[future_mask].copy()
    future["Month"] = future["ds"].dt.strftime("%b %Y")
    future = future[["Month", "forecast", "ci_lower", "ci_upper"]].rename(
        columns={"forecast": "Forecast", "ci_lower": "Lower CI", "ci_upper": "Upper CI"})
    st.dataframe(
        future.style.format({"Forecast":"${:,.0f}", "Lower CI":"${:,.0f}", "Upper CI":"${:,.0f}"}),
        use_container_width=True, hide_index=True
    )

    # ── KPI Tracker ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("KPI Alert Monitor")

    window = st.selectbox("Rolling window (months)", [1, 2, 3, 6], index=2)
    alerts, history = get_kpi_data(df, window)

    # Status cards
    STATUS_COLORS = kpi.STATUS_COLORS
    STATUS_EMOJI  = kpi.STATUS_EMOJI
    cols = st.columns(len(alerts))
    for col, alert in zip(cols, alerts):
        color = STATUS_COLORS[alert["status"]]
        css_class = f"alert-{alert['status'].lower()}"
        col.markdown(f"""
        <div class="metric-card {css_class}">
          <div style="font-size:1.6rem;font-weight:700;color:{color}">{alert['display_value']}</div>
          <div style="font-size:0.8rem;color:#374151;margin-top:4px">{alert['label']}</div>
          <div style="font-size:0.8rem;color:#6B7280">Target: {alert['display_target']}</div>
          <div style="font-size:0.85rem;font-weight:600;color:{color};margin-top:6px">
            {STATUS_EMOJI[alert['status']]} {alert['status']}  {alert['pct_vs_target']:+.1f}%
          </div>
        </div>""", unsafe_allow_html=True)

    # KPI trend sparklines
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("KPI Trends Over Time")
    cols2 = st.columns(len(alerts))
    x_labels = history["year_month"].tolist()
    step = max(1, len(x_labels) // 6)

    for col, alert in zip(cols2, alerts):
        key    = alert["key"]
        vals   = history[key].values
        target = kpi.KPI_CONFIG[key]["target"]

        fig, ax = plt.subplots(figsize=(3.5, 2), facecolor=BG)
        statuses = history[f"{key}_status"].values
        point_colors = [STATUS_COLORS[s] for s in statuses]
        ax.plot(range(len(vals)), vals, color="#CBD5E1", linewidth=1)
        ax.scatter(range(len(vals)), vals, c=point_colors, s=18,
                   zorder=2, edgecolors="white", linewidth=0.3)
        ax.axhline(target, color=AMB, linewidth=1, linestyle="--", alpha=0.8)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels(x_labels[::step], rotation=45, ha="right", fontsize=5.5)
        cfg = kpi.KPI_CONFIG[key]
        if cfg["unit"] == "$":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, _: f"${x:,.0f}" if x < 10_000 else f"${x/1e3:.0f}K"))
        elif cfg["unit"] == "%":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        _style(ax, alert["label"])
        col.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""<div class="insight-box">
     <strong>Forecast insight:</strong> The model projects moderate revenue growth
    over the next 6 months, with seasonal peaks in Q4. KPI monitoring shows all metrics
    currently in GREEN — set up weekly scheduled runs to catch dips early.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    df_raw = load_data()
    df     = sidebar_filters(df_raw)

    if len(df) == 0:
        st.warning("No data matches the selected filters. Please adjust the sidebar.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Trends",
        "Products",
        "Segments",
        "Forecast & KPIs",
    ])

    with tab1: tab_overview(df)
    with tab2: tab_trends(df)
    with tab3: tab_products(df)
    with tab4: tab_segments(df)
    with tab5: tab_forecast_kpi(df)


if __name__ == "__main__":
    main()
