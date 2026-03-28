"""
src/kpi_tracker.py
------------------
Automated KPI monitoring system with threshold-based alerting.

WHY this matters for a resume:
  Most junior analysts just plot data. A KPI tracker shows you
  think about *operationalising* analysis — turning insights into
  automated decisions. This maps directly to BI/Analytics Engineer roles.

Interview talking point:
  "I built a threshold-based alert system so stakeholders get notified
  automatically when key metrics deviate — no manual checking required.
  In production, this would integrate with Slack or email via a scheduler."

Architecture:
  - Define KPIs as config dicts (thresholds fully parameterised)
  - Compare actuals vs benchmarks
  - Classify each KPI as: GREEN / YELLOW / RED
  - Output: console alerts + alert_log dict + dashboard-ready data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime


# ── KPI definitions ────────────────────────────────────────────────────────────
# Thresholds are business rules — document them so anyone can change them.
KPI_CONFIG = {
    "monthly_revenue": {
        "label":      "Monthly Revenue",
        "unit":       "$",
        "target":     145_000,   # Example target (adjust per dataset)
        "warn_pct":   0.85,      # < 85% of target → YELLOW
        "crit_pct":   0.70,      # < 70% of target → RED
        "higher_is_better": True,
    },
    "avg_order_value": {
        "label":      "Avg Order Value",
        "unit":       "$",
        "target":     420,
        "warn_pct":   0.90,
        "crit_pct":   0.75,
        "higher_is_better": True,
    },
    "profit_margin": {
        "label":      "Profit Margin %",
        "unit":       "%",
        "target":     0.26,      # 26%
        "warn_pct":   0.88,
        "crit_pct":   0.75,
        "higher_is_better": True,
    },
    "order_count": {
        "label":      "Monthly Orders",
        "unit":       "",
        "target":     330,
        "warn_pct":   0.85,
        "crit_pct":   0.70,
        "higher_is_better": True,
    },
    "discount_rate": {
        "label":      "Avg Discount Rate",
        "unit":       "%",
        "target":     0.10,      # 10% — cap, not floor
        "warn_pct":   1.20,      # > 120% of target → YELLOW (too high)
        "crit_pct":   1.50,      # > 150% of target → RED
        "higher_is_better": False,
    },
}

STATUS_COLORS = {
    "GREEN":  "#16A34A",
    "YELLOW": "#D97706",
    "RED":    "#DC2626",
}

STATUS_EMOJI = {"GREEN": "✓", "YELLOW": "⚠", "RED": "✗"}


# ── Compute KPIs from raw data ─────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, window_months: int = 3) -> dict:
    """
    Compute current KPI values using the most recent N months of data.
    WHY rolling window: avoids misleading comparisons against a partial month.

    Returns: dict of {kpi_key: current_value}
    """
    latest = df["date"].max()
    cutoff  = latest - pd.DateOffset(months=window_months)
    recent  = df[df["date"] >= cutoff]

    n_months = max(recent["year_month"].nunique(), 1)

    kpis = {
        "monthly_revenue": recent["revenue"].sum() / n_months,
        "avg_order_value": recent["revenue"].mean(),
        "profit_margin":   recent["profit_margin"].mean(),
        "order_count":     len(recent) / n_months,
        "discount_rate":   recent["discount"].mean(),
    }
    return kpis


# ── Alert engine ───────────────────────────────────────────────────────────────

def classify_kpi(key: str, value: float) -> dict:
    """
    Compare value to thresholds and assign GREEN / YELLOW / RED.
    Returns full alert record for logging and display.
    """
    cfg = KPI_CONFIG[key]
    target = cfg["target"]
    ratio  = value / target if target != 0 else 1.0

    if cfg["higher_is_better"]:
        if ratio >= cfg["warn_pct"]:   status = "GREEN"
        elif ratio >= cfg["crit_pct"]: status = "YELLOW"
        else:                          status = "RED"
    else:
        # Lower is better (e.g. discount rate)
        if ratio <= cfg["warn_pct"]:   status = "GREEN"
        elif ratio <= cfg["crit_pct"]: status = "YELLOW"
        else:                          status = "RED"

    pct_vs_target = (value - target) / target * 100

    # Format value for display
    if cfg["unit"] == "$":
        display_val    = f"${value:,.2f}"
        display_target = f"${target:,.2f}"
    elif cfg["unit"] == "%":
        display_val    = f"{value:.1%}"
        display_target = f"{target:.1%}"
    else:
        display_val    = f"{value:,.1f}"
        display_target = f"{target:,.1f}"

    return {
        "key":             key,
        "label":           cfg["label"],
        "value":           value,
        "target":          target,
        "status":          status,
        "pct_vs_target":   pct_vs_target,
        "display_value":   display_val,
        "display_target":  display_target,
        "ratio":           ratio,
        "higher_is_better": cfg["higher_is_better"],
    }


def run_kpi_alerts(df: pd.DataFrame, window_months: int = 3) -> list[dict]:
    """
    Run the full alert pipeline. Returns a list of alert records.

    This function is the one you'd call from a scheduled job (cron / Airflow).
    """
    kpi_values = compute_kpis(df, window_months)
    alerts     = [classify_kpi(k, v) for k, v in kpi_values.items()]

    # Console output
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n── KPI Alert Report ({timestamp}) ─────────────────────────────")
    print(f"  {'KPI':<25} {'Current':>12} {'Target':>12} {'vs Target':>10}  Status")
    print("  " + "─" * 70)
    for a in alerts:
        icon = STATUS_EMOJI[a["status"]]
        print(f"  {a['label']:<25} {a['display_value']:>12} "
              f"{a['display_target']:>12} {a['pct_vs_target']:>+9.1f}%  "
              f"[{icon}] {a['status']}")

    # Summary
    reds    = [a for a in alerts if a["status"] == "RED"]
    yellows = [a for a in alerts if a["status"] == "YELLOW"]
    print(f"\n  {'─'*70}")
    print(f"  Summary: {len(reds)} RED  |  {len(yellows)} YELLOW  |  "
          f"{len(alerts) - len(reds) - len(yellows)} GREEN")
    if reds:
        print(f"\n  [CRITICAL] Action required on: {', '.join(a['label'] for a in reds)}")
    print("─" * 72)

    return alerts


# ── Historical KPI tracking ────────────────────────────────────────────────────

def build_kpi_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute KPIs for every month (rolling 3-month window).
    Used to plot trends in the dashboard.

    Returns: long-format DataFrame (month, kpi_key, value)
    """
    months = sorted(df["year_month"].unique())
    records = []

    for ym in months:
        # Use data up to and including this month
        mask = df["year_month"] <= ym
        subset = df[mask].copy()

        # Only use last 3 months of data
        last3 = sorted(subset["year_month"].unique())[-3:]
        subset = subset[subset["year_month"].isin(last3)]

        n_months = max(len(last3), 1)
        records.append({
            "year_month":    ym,
            "monthly_revenue": subset["revenue"].sum() / n_months,
            "avg_order_value": subset["revenue"].mean(),
            "profit_margin":   subset["profit_margin"].mean(),
            "order_count":     len(subset) / n_months,
            "discount_rate":   subset["discount"].mean(),
        })

    history = pd.DataFrame(records)
    # Add status classifications
    for key in KPI_CONFIG:
        history[f"{key}_status"] = history[key].apply(
            lambda v: classify_kpi(key, v)["status"]
        )
    return history


# ── Dashboard chart ────────────────────────────────────────────────────────────

def plot_kpi_dashboard(alerts: list[dict],
                       history: pd.DataFrame,
                       output_dir: str = "outputs/charts") -> str:
    """
    2-row layout:
    Row 1: KPI status cards (gauge-style)
    Row 2: Historical trend for each KPI with target line
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(alerts)

    fig = plt.figure(figsize=(18, 10), facecolor="#F8FAFC")
    fig.suptitle("KPI Monitoring Dashboard", fontsize=16,
                 fontweight="bold", color="#1E293B", y=1.01)

    # ── Row 1: Status cards ──────────────────────────────────────────────
    for i, alert in enumerate(alerts):
        ax = fig.add_subplot(2, n, i + 1)
        color = STATUS_COLORS[alert["status"]]

        # Background card
        ax.set_facecolor(color + "18")   # very light tint
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        ax.text(0.5, 0.75, alert["display_value"],
                transform=ax.transAxes, ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)
        ax.text(0.5, 0.50, alert["label"],
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#1E293B")
        ax.text(0.5, 0.28, f"Target: {alert['display_target']}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#64748B")
        pct_str = f"{alert['pct_vs_target']:+.1f}% vs target"
        ax.text(0.5, 0.12, pct_str,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=color, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Status badge
        badge_color = STATUS_COLORS[alert["status"]]
        ax.text(0.92, 0.92, STATUS_EMOJI[alert["status"]],
                transform=ax.transAxes, fontsize=12,
                color=badge_color, ha="right", va="top", fontweight="bold")

    # ── Row 2: KPI trend lines ───────────────────────────────────────────
    x_labels = history["year_month"].tolist()
    step = max(1, len(x_labels) // 8)

    for i, alert in enumerate(alerts):
        ax = fig.add_subplot(2, n, n + i + 1)
        key = alert["key"]
        vals = history[key].values
        target = KPI_CONFIG[key]["target"]

        # Color each point by its alert status
        statuses = history[f"{key}_status"].values
        point_colors = [STATUS_COLORS[s] for s in statuses]

        ax.plot(range(len(vals)), vals, color="#94A3B8",
                linewidth=1.2, zorder=1)
        ax.scatter(range(len(vals)), vals, c=point_colors,
                   s=22, zorder=2, edgecolors="white", linewidth=0.4)
        ax.axhline(target, color="#D97706", linewidth=1.2,
                   linestyle="--", alpha=0.8, label="Target")

        # Shade alert zones
        warn_line = target * KPI_CONFIG[key]["warn_pct"]
        crit_line = target * KPI_CONFIG[key]["crit_pct"]
        y_min = min(vals.min(), crit_line) * 0.95
        y_max = max(vals.max(), target) * 1.05

        if KPI_CONFIG[key]["higher_is_better"]:
            ax.axhspan(y_min, crit_line, color="#DC2626", alpha=0.06)
            ax.axhspan(crit_line, warn_line, color="#D97706", alpha=0.06)
        else:
            ax.axhspan(warn_line, y_max, color="#D97706", alpha=0.06)
            ax.axhspan(target * KPI_CONFIG[key]["crit_pct"], y_max, color="#DC2626", alpha=0.06)

        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels(x_labels[::step], rotation=45, ha="right", fontsize=6)
        ax.set_title(alert["label"], fontsize=9, fontweight="bold", color="#1E293B")
        ax.set_facecolor("#F8FAFC")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.6, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#64748B", labelsize=7)

        # Format y-axis
        cfg = KPI_CONFIG[key]
        if cfg["unit"] == "$":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, _: f"${x:,.0f}" if x < 10_000 else f"${x/1e3:.0f}K"))
        elif cfg["unit"] == "%":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Legend for Row 2
    patches = [mpatches.Patch(color=STATUS_COLORS[s], label=s)
               for s in ["GREEN", "YELLOW", "RED"]]
    patches.append(mpatches.Patch(color="#D97706", label="Target", linestyle="--", fill=False))
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.7, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    filepath = os.path.join(output_dir, "12_kpi_dashboard.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    print(f"  ✓ Saved → {filepath}")
    return filepath


# ── Runner ────────────────────────────────────────────────────────────────────

def run_kpi_tracker(df: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Entry point. Returns alerts list + history DataFrame."""
    print("\n[KPI Tracker] Running alert checks…")
    alerts  = run_kpi_alerts(df)
    history = build_kpi_history(df)
    plot_kpi_dashboard(alerts, history)
    return alerts, history
