"""
src/visualization.py
--------------------
All charts built with pure Matplotlib.
Each function: takes a DataFrame → saves a PNG → returns the filepath.

Design: Separated from EDA so you can swap chart libraries later
        without touching business logic.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = "outputs/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#2563EB",   # Blue
    "secondary": "#16A34A",   # Green
    "accent":    "#DC2626",   # Red
    "warning":   "#D97706",   # Amber
    "purple":    "#7C3AED",
    "bg":        "#F8FAFC",
    "grid":      "#E2E8F0",
    "text":      "#1E293B",
    "muted":     "#64748B",
}

CAT_PALETTE  = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
                "#0891B2", "#BE185D", "#059669", "#EA580C", "#4338CA"]

REGION_COLORS = {
    "West": "#2563EB", "East": "#16A34A",
    "Central": "#D97706", "South": "#DC2626"
}


def _style_axes(ax, title, xlabel="", ylabel=""):
    """Common styling applied to every chart."""
    ax.set_title(title, fontsize=14, fontweight="bold",
                 color=COLORS["text"], pad=14)
    ax.set_xlabel(xlabel, fontsize=10, color=COLORS["muted"])
    ax.set_ylabel(ylabel, fontsize=10, color=COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_edgecolor(COLORS["grid"])


def _save(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  ✓ Saved → {filepath}")
    return filepath


# ── Chart 1: Sales & Profit Trend (Line) ─────────────────────────────────────

def plot_sales_trend(trend_df):
    """
    Dual-line chart: monthly revenue + profit.
    WHY dual-axis: Revenue and profit have different scales; overlaying
    them on one axis compresses profit and hides margin insights.
    """
    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor=COLORS["bg"])

    x = range(len(trend_df))
    labels = trend_df["year_month"].tolist()

    ax1.plot(x, trend_df["revenue"], color=COLORS["primary"],
             linewidth=2, marker="o", markersize=3, label="Revenue")
    ax1.fill_between(x, trend_df["revenue"], alpha=0.08, color=COLORS["primary"])

    ax2 = ax1.twinx()
    ax2.plot(x, trend_df["profit"], color=COLORS["secondary"],
             linewidth=1.8, linestyle="--", marker="s", markersize=3, label="Profit")
    ax2.set_ylabel("Profit ($)", fontsize=10, color=COLORS["muted"])
    ax2.tick_params(colors=COLORS["muted"], labelsize=9)
    ax2.spines[["top", "right"]].set_edgecolor(COLORS["grid"])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    # X-axis: show every 3rd label to avoid clutter
    step = max(1, len(labels) // 12)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    _style_axes(ax1, "Monthly Revenue & Profit Trend", ylabel="Revenue ($)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9, framealpha=0.7)

    return _save(fig, "01_sales_trend.png")


# ── Chart 2: Top 10 Products (Horizontal Bar) ─────────────────────────────────

def plot_top_products(products_df):
    """
    Horizontal bar preferred over vertical when labels are long strings.
    Color-coded by rank intensity.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    df = products_df.sort_values("revenue")

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df["product"], df["revenue"], color=colors, edgecolor="white")

    # Value labels on bars
    for bar, val in zip(bars, df["revenue"]):
        ax.text(bar.get_width() + df["revenue"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", ha="left",
                fontsize=8, color=COLORS["text"])

    ax.set_xlim(0, df["revenue"].max() * 1.18)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    _style_axes(ax, "Top 10 Products by Revenue", xlabel="Revenue ($)")

    return _save(fig, "02_top_products.png")


# ── Chart 3: Category Revenue Share (Pie) ─────────────────────────────────────

def plot_category_pie(category_df):
    """
    Pie chart for part-to-whole relationships.
    Explode the top slice for emphasis (a classic hiring-manager pleaser).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor=COLORS["bg"])

    # Pie
    explode = [0.04 if i == 0 else 0 for i in range(len(category_df))]
    wedges, texts, autotexts = ax1.pie(
        category_df["revenue"],
        labels=category_df["category"],
        autopct="%1.1f%%",
        explode=explode,
        colors=CAT_PALETTE[:len(category_df)],
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.set_title("Category Revenue Share", fontsize=14,
                  fontweight="bold", color=COLORS["text"], pad=14)

    # Companion bar (margin)
    colors = CAT_PALETTE[:len(category_df)]
    bars = ax2.bar(category_df["category"], category_df["avg_margin"] * 100,
                   color=colors, edgecolor="white", width=0.5)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=9, color=COLORS["text"])
    _style_axes(ax2, "Avg Profit Margin by Category", ylabel="Margin (%)")
    ax2.set_ylim(0, category_df["avg_margin"].max() * 100 * 1.25)

    fig.suptitle("Category Performance Overview", fontsize=16,
                 fontweight="bold", color=COLORS["text"], y=1.02)

    return _save(fig, "03_category_overview.png")


# ── Chart 4: Region Performance (Grouped Bar) ─────────────────────────────────

def plot_region_performance(region_df):
    """
    Grouped bars: revenue vs profit per region.
    WHY grouped: Allows direct visual comparison of margin health across regions.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    x = np.arange(len(region_df))
    w = 0.35

    r1 = ax.bar(x - w / 2, region_df["revenue"], w, label="Revenue",
                color=COLORS["primary"], edgecolor="white")
    r2 = ax.bar(x + w / 2, region_df["profit"],  w, label="Profit",
                color=COLORS["secondary"], edgecolor="white")

    for bars in [r1, r2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + region_df["revenue"].max() * 0.005,
                    f"${bar.get_height()/1e3:.0f}K",
                    ha="center", va="bottom", fontsize=8, color=COLORS["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(region_df["region"], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax.legend(fontsize=9)
    _style_axes(ax, "Revenue & Profit by Region", ylabel="Amount ($)")

    return _save(fig, "04_region_performance.png")


# ── Chart 5: Order Value Distribution (Histogram) ─────────────────────────────

def plot_order_distribution(revenue_series):
    """
    Histogram with KDE overlay.
    WHY: Reveals order-size clusters → informs pricing tiers and upsell strategy.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS["bg"])

    # Clip extreme outliers for display (>99th percentile)
    cap = revenue_series.quantile(0.99)
    data = revenue_series[revenue_series <= cap]

    n, bins, patches = ax.hist(data, bins=50, color=COLORS["primary"],
                                edgecolor="white", alpha=0.85)

    # Colour bars by decile for gradient effect
    for frac, patch in zip(np.linspace(0.3, 1.0, len(patches)), patches):
        patch.set_facecolor(plt.cm.Blues(frac))

    # Vertical lines for mean and median
    ax.axvline(data.mean(),   color=COLORS["accent"],  linestyle="--",
               linewidth=1.5, label=f"Mean  ${data.mean():,.0f}")
    ax.axvline(data.median(), color=COLORS["warning"], linestyle="-.",
               linewidth=1.5, label=f"Median ${data.median():,.0f}")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    _style_axes(ax, "Order Value Distribution", xlabel="Order Revenue ($)", ylabel="Frequency")

    return _save(fig, "05_order_distribution.png")


# ── Chart 6: Monthly Seasonality (Bar) ────────────────────────────────────────

def plot_monthly_seasonality(monthly_df):
    """
    Average revenue per calendar month across all years.
    Highlights Q4 spike — the core business insight.
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS["bg"])

    peak_idx = monthly_df["avg_revenue"].idxmax()
    bar_colors = [
        COLORS["accent"] if i == peak_idx else COLORS["primary"]
        for i in monthly_df.index
    ]
    bars = ax.bar(monthly_df["month_name"].astype(str),
                  monthly_df["avg_revenue"],
                  color=bar_colors, edgecolor="white", width=0.6)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + monthly_df["avg_revenue"].max() * 0.01,
                f"${bar.get_height()/1e3:.1f}K",
                ha="center", va="bottom", fontsize=8, color=COLORS["text"])

    peak_label = mpatches.Patch(color=COLORS["accent"], label="Peak Month")
    base_label  = mpatches.Patch(color=COLORS["primary"], label="Other Months")
    ax.legend(handles=[peak_label, base_label], fontsize=9)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    _style_axes(ax, "Avg Monthly Revenue — Seasonal Pattern",
                xlabel="Month", ylabel="Avg Revenue ($)")

    return _save(fig, "06_monthly_seasonality.png")


# ── Chart 7: Heatmap (Month × Region) ─────────────────────────────────────────

def plot_heatmap(pivot_df):
    """
    Custom Matplotlib heatmap — no Seaborn needed.
    Encodes revenue intensity in colour; adds text annotations.
    """
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLORS["bg"])
    data  = pivot_df.values.astype(float)
    cmap  = plt.cm.YlOrRd

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > data.max() * 0.6 else COLORS["text"]
            ax.text(j, i, f"${val/1e3:.0f}K",
                    ha="center", va="center", fontsize=8,
                    color=text_color, fontweight="bold")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, fontsize=10)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=9)
    ax.set_title("Revenue Heatmap — Month × Region",
                 fontsize=14, fontweight="bold", color=COLORS["text"], pad=14)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Revenue ($)", fontsize=9, color=COLORS["muted"])
    cbar.ax.tick_params(labelsize=8)

    return _save(fig, "07_heatmap_month_region.png")


# ── Chart 8: Top 10 Customers (Horizontal Bar) ───────────────────────────────

def plot_top_customers(customers_df):
    """Identifies VIP customers — great for account-management discussions."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    df = customers_df.sort_values("total_revenue")

    colors = plt.cm.Greens(np.linspace(0.35, 0.9, len(df)))
    bars = ax.barh(df["customer_id"], df["total_revenue"],
                   color=colors, edgecolor="white")

    for bar, orders in zip(bars, df["total_orders"]):
        ax.text(bar.get_width() + df["total_revenue"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"${bar.get_width():,.0f}  ({orders} orders)",
                va="center", ha="left", fontsize=8, color=COLORS["text"])

    ax.set_xlim(0, df["total_revenue"].max() * 1.28)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _style_axes(ax, "Top 10 Customers by Revenue", xlabel="Total Revenue ($)")

    return _save(fig, "08_top_customers.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def generate_all_charts(df, eda):
    """
    Convenience function called from the notebook.
    Passes correct data slices to each chart function.
    """
    print("\nGenerating charts…")
    paths = []

    paths.append(plot_sales_trend(eda.monthly_sales_trend(df)))
    paths.append(plot_top_products(eda.top_products(df, 10)))
    paths.append(plot_category_pie(eda.category_performance(df)))
    paths.append(plot_region_performance(eda.region_performance(df)))
    paths.append(plot_order_distribution(eda.order_value_distribution(df)))
    paths.append(plot_monthly_seasonality(eda.monthly_avg_revenue(df)))
    paths.append(plot_heatmap(eda.heatmap_data(df)))
    paths.append(plot_top_customers(eda.top_customers(df, 10)))

    print(f"\n✓ {len(paths)} charts saved to '{OUTPUT_DIR}/'")
    return paths
