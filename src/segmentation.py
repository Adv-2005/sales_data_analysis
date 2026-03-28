"""
src/segmentation.py
-------------------
RFM (Recency, Frequency, Monetary) Customer Segmentation.

WHY RFM?
  It's the industry-standard framework for customer analytics.
  Every CRM tool (Klaviyo, HubSpot, Salesforce) has some RFM equivalent.
  Interviewers will recognise it immediately.

HOW IT WORKS:
  1. Score each customer on 3 dimensions (1–4 scale, quartile-based)
  2. Combine scores into an RFM string e.g. "444" = Champion
  3. Map score combos to named segments
  4. Derive business actions per segment

Interview talking point:
  "I used quartile-based scoring so the model adapts to any dataset
  without needing manually tuned thresholds."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# ── Segment definitions ───────────────────────────────────────────────────────
# Each segment: (label, description, business action, hex colour)
SEGMENTS = {
    "Champions":          ("Champions",         "Bought recently, buy often, spend the most",
                           "Reward them. They can be early adopters for new products.",       "#16A34A"),
    "Loyal Customers":    ("Loyal Customers",   "Buy regularly; respond well to loyalty schemes",
                           "Upsell higher-value products. Ask for reviews.",                   "#2563EB"),
    "Potential Loyalist": ("Potential Loyalist","Recent customers with average frequency",
                           "Offer membership / loyalty programmes.",                           "#7C3AED"),
    "New Customers":      ("New Customers",     "Bought recently but not often",
                           "Onboard well. Provide start guides and first-purchase offers.",    "#0891B2"),
    "Promising":          ("Promising",         "Recent shoppers but haven't spent much",
                           "Build brand awareness, offer free trials or samples.",             "#65A30D"),
    "Need Attention":     ("Need Attention",    "Above-average R, F, M but declining",
                           "Limited-time offers. Ask if anything is wrong.",                   "#D97706"),
    "About to Sleep":     ("About to Sleep",   "Below-average recency and frequency",
                           "Share relevant resources. Offer discounts.",                       "#EA580C"),
    "At Risk":            ("At Risk",           "Spent big, bought often — but long ago",
                           "Send personalised reactivation emails.",                           "#DC2626"),
    "Cannot Lose Them":   ("Cannot Lose Them", "Made big purchases, haven't returned",
                           "Win them back via renewals or newer products.",                    "#9F1239"),
    "Hibernating":        ("Hibernating",       "Last purchase long ago, low spend",
                           "Offer relevant products and special discounts.",                   "#78716C"),
    "Lost":               ("Lost",              "Lowest recency, frequency, monetary",
                           "Revive interest with reach-out campaign; otherwise deprioritise.", "#475569"),
}


def _assign_segment(rfm_score: str) -> str:
    """
    Map an RFM string to a named segment.
    Logic mirrors the widely-used RFM segmentation matrix.
    """
    r = int(rfm_score[0])
    f = int(rfm_score[1])
    m = int(rfm_score[2])

    if r >= 4 and f >= 4:                          return "Champions"
    if r >= 3 and f >= 3:                          return "Loyal Customers"
    if r >= 4 and f <= 2:                          return "New Customers"
    if r >= 3 and f == 2:                          return "Potential Loyalist"
    if r == 3 and f == 1:                          return "Promising"
    if r == 2 and f >= 3:                          return "Need Attention"
    if r == 2 and f <= 2 and m >= 3:               return "About to Sleep"
    if r <= 2 and f >= 3 and m >= 3:               return "At Risk"
    if r == 1 and f >= 4 and m >= 4:               return "Cannot Lose Them"
    if r == 2 and f <= 2 and m <= 2:               return "Hibernating"
    return "Lost"


# ── Core RFM computation ──────────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Compute R, F, M for every customer.

    Args:
        df           : cleaned sales DataFrame
        snapshot_date: treat this as "today"; defaults to max(date) + 1 day

    Returns: customer-level DataFrame with R, F, M values + scores + segment
    """
    if snapshot_date is None:
        snapshot = df["date"].max() + pd.Timedelta(days=1)
    else:
        snapshot = pd.Timestamp(snapshot_date)

    # Raw RFM values
    rfm = (
        df.groupby("customer_id")
          .agg(
              recency   =("date",     lambda x: (snapshot - x.max()).days),
              frequency =("order_id", "count"),
              monetary  =("revenue",  "sum"),
          )
          .reset_index()
    )

    # Quartile scoring (1 = worst, 4 = best)
    # Recency: lower days = better → reverse rank
    rfm["r_score"] = pd.qcut(rfm["recency"],   q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"],  q=4, labels=[1, 2, 3, 4]).astype(int)

    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    rfm["rfm_total"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    rfm["segment"]   = rfm["rfm_score"].apply(_assign_segment)

    return rfm


# ── Segment summary ───────────────────────────────────────────────────────────

def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Business-level summary per segment.
    Joins action recommendations from SEGMENTS dict.
    """
    summary = (
        rfm.groupby("segment")
           .agg(
               customers=("customer_id", "count"),
               avg_recency=("recency",    "mean"),
               avg_frequency=("frequency","mean"),
               avg_monetary=("monetary",  "mean"),
               total_revenue=("monetary", "sum"),
           )
           .reset_index()
    )
    summary["avg_recency"]   = summary["avg_recency"].round(0).astype(int)
    summary["avg_frequency"] = summary["avg_frequency"].round(1)
    summary["avg_monetary"]  = summary["avg_monetary"].round(2)
    summary["pct_customers"] = (summary["customers"] / summary["customers"].sum() * 100).round(1)

    # Attach actions
    summary["action"] = summary["segment"].map(
        lambda s: SEGMENTS.get(s, ("", "", "No action defined", "#999"))[2]
    )
    return summary.sort_values("total_revenue", ascending=False)


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_rfm_segments(rfm: pd.DataFrame, output_dir: str = "outputs/charts") -> str:
    """
    3-panel chart:
    1. Treemap-style bar: customer count per segment
    2. Avg monetary per segment
    3. RFM scatter: recency vs frequency, sized by monetary
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = segment_summary(rfm)

    fig = plt.figure(figsize=(18, 6), facecolor="#F8FAFC")
    fig.suptitle("RFM Customer Segmentation", fontsize=16,
                 fontweight="bold", color="#1E293B", y=1.01)

    # ── Panel 1: Customer count ─────────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    colors = [SEGMENTS.get(s, ("","","","#888"))[3] for s in summary["segment"]]
    bars = ax1.barh(summary["segment"], summary["customers"],
                    color=colors, edgecolor="white", height=0.6)
    for bar, pct in zip(bars, summary["pct_customers"]):
        ax1.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 f"{pct}%", va="center", fontsize=7.5, color="#1E293B")
    ax1.set_xlim(0, summary["customers"].max() * 1.3)
    ax1.set_title("Customers per Segment", fontsize=11, fontweight="bold", color="#1E293B")
    ax1.set_facecolor("#F8FAFC")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=8, colors="#64748B")

    # ── Panel 2: Avg revenue per segment ────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.barh(summary["segment"], summary["avg_monetary"],
             color=colors, edgecolor="white", height=0.6)
    ax2.set_title("Avg Revenue per Customer", fontsize=11, fontweight="bold", color="#1E293B")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.set_facecolor("#F8FAFC")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8, colors="#64748B")
    ax2.set_yticklabels([])

    # ── Panel 3: Scatter (R vs F, bubble = M) ───────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    seg_colors_map = {s: SEGMENTS.get(s, ("","","","#888"))[3] for s in rfm["segment"].unique()}
    scatter_colors = [seg_colors_map.get(s, "#888") for s in rfm["segment"]]
    sizes = ((rfm["monetary"] - rfm["monetary"].min())
             / (rfm["monetary"].max() - rfm["monetary"].min()) * 200 + 10)

    ax3.scatter(rfm["recency"], rfm["frequency"], c=scatter_colors,
                s=sizes, alpha=0.4, edgecolors="white", linewidth=0.3)

    # Legend patches
    patches = [mpatches.Patch(color=SEGMENTS[s][3], label=s)
               for s in summary["segment"] if s in SEGMENTS]
    ax3.legend(handles=patches, fontsize=6, loc="upper right",
               ncol=1, framealpha=0.6, markerscale=0.7)
    ax3.set_xlabel("Recency (days)", fontsize=9, color="#64748B")
    ax3.set_ylabel("Frequency (orders)", fontsize=9, color="#64748B")
    ax3.set_title("Recency vs Frequency\n(bubble = revenue)", fontsize=11,
                  fontweight="bold", color="#1E293B")
    ax3.set_facecolor("#F8FAFC")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(labelsize=8, colors="#64748B")
    ax3.grid(color="#E2E8F0", linewidth=0.6, linestyle="--")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "09_rfm_segmentation.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    print(f"  ✓ Saved → {filepath}")
    return filepath


def plot_rfm_heatmap(rfm: pd.DataFrame, output_dir: str = "outputs/charts") -> str:
    """
    2D heatmap of avg monetary value across R-score × F-score grid.
    WHY: Shows the value of each score combination at a glance.
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = (
        rfm.groupby(["r_score", "f_score"])["monetary"]
           .mean()
           .unstack(fill_value=0)
           .astype(float)          # ensure numeric after unstack
    )

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#F8FAFC")
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"${val:,.0f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if val > pivot.values.max() * 0.55 else "#1E293B",
                    fontweight="bold")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"F={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"R={r}" for r in pivot.index])
    ax.set_title("Avg Revenue by RFM Score Grid\n(R=4 = most recent, F=4 = most frequent)",
                 fontsize=12, fontweight="bold", color="#1E293B", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Avg Revenue ($)", fontsize=9)

    filepath = os.path.join(output_dir, "10_rfm_heatmap.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    print(f"  ✓ Saved → {filepath}")
    return filepath


# ── Segment report ────────────────────────────────────────────────────────────

def print_segment_actions(rfm: pd.DataFrame) -> None:
    """Console-friendly action table — great for report.txt."""
    summary = segment_summary(rfm)
    print("\n── RFM Segment Actions ─────────────────────────────────────────")
    print(f"{'Segment':<22} {'Customers':>9} {'Avg $':>9}  Action")
    print("─" * 80)
    for _, row in summary.iterrows():
        print(f"  {row['segment']:<20} {row['customers']:>9,} {row['avg_monetary']:>9,.0f}  {row['action'][:55]}")
    print("─" * 80)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Call this from run_analysis.py or Streamlit."""
    print("\n[Segmentation] Computing RFM scores…")
    rfm = compute_rfm(df)
    print(f"  Segments found: {sorted(rfm['segment'].unique())}")
    print_segment_actions(rfm)
    plot_rfm_segments(rfm)
    plot_rfm_heatmap(rfm)
    return rfm
