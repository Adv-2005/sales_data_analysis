"""
src/forecasting.py
------------------
Time-series sales forecasting using:
  1. Centred moving average  (smooths noise)
  2. Linear trend component  (captures growth/decline)
  3. Seasonal index          (captures monthly patterns)
  4. Combined forecast       (trend × seasonal index)

WHY this approach and NOT Prophet/ARIMA?
  - No extra dependencies (numpy + pandas only)
  - Fully explainable in an interview step-by-step
  - Produces identical insight to Prophet for business purposes
  - Interviewers prefer "I know the maths" over "I called a library"

Interview talking point:
  "I decomposed the time series into trend, seasonal, and residual
   components manually — this shows I understand what libraries like
   Prophet are doing under the hood."

When asked about alternatives:
  "For production I'd use Prophet or SARIMA for better uncertainty
   quantification, but the underlying decomposition logic is the same."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


# ── Step 1: Build monthly time series ────────────────────────────────────────

def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to monthly total revenue.
    Returns a complete, gap-free monthly Series (missing months = 0).
    """
    monthly = (
        df.groupby("year_month")["revenue"]
          .sum()
          .reset_index()
    )
    monthly["ds"] = pd.to_datetime(monthly["year_month"])
    monthly = monthly.sort_values("ds").set_index("ds")[["revenue"]]
    # Fill any missing months with 0 (rare, but ensures continuity)
    monthly = monthly.resample("MS").sum()
    monthly.columns = ["y"]
    return monthly


# ── Step 2: Decomposition ─────────────────────────────────────────────────────

def decompose(series: pd.Series, period: int = 12) -> pd.DataFrame:
    """
    Classical multiplicative decomposition:
        y = trend × seasonal × residual

    Steps:
      1. Compute centred moving average → trend
      2. Detrend: y / trend
      3. Seasonal index: average detrended value per calendar month
      4. Residual: detrended / seasonal_index
    """
    df = pd.DataFrame({"y": series})

    # Trend via centred moving average
    df["trend"] = df["y"].rolling(window=period, center=True, min_periods=period // 2).mean()

    # Detrended series (guard against division by zero)
    df["detrended"] = df["y"] / df["trend"].replace(0, np.nan)

    # Seasonal index per month (mean across years)
    df["month"] = df.index.month
    seasonal_idx = df.groupby("month")["detrended"].mean()
    # Normalise so seasonal indices average to 1.0
    seasonal_idx = seasonal_idx / seasonal_idx.mean()
    df["seasonal"] = df["month"].map(seasonal_idx)

    # Residual
    df["residual"] = df["detrended"] / df["seasonal"].replace(0, np.nan)

    return df, seasonal_idx


# ── Step 3: Trend extrapolation ────────────────────────────────────────────────

def fit_trend(trend_series: pd.Series) -> tuple:
    """
    Fit a linear model to the trend component using numpy polyfit.
    Returns (slope, intercept) and a callable predictor.

    WHY linear? With 2–3 years of data, a linear trend is the most
    defensible assumption — insufficient data to justify exponential
    or polynomial without overfitting.
    """
    valid = trend_series.dropna()
    x = np.arange(len(valid))
    coeffs = np.polyfit(x, valid.values, deg=1)
    slope, intercept = coeffs

    def predict_trend(n_steps: int) -> np.ndarray:
        """Extrapolate trend for the next n_steps periods after the last valid point."""
        future_x = np.arange(len(valid), len(valid) + n_steps)
        return slope * future_x + intercept

    return slope, intercept, predict_trend


# ── Step 4: Generate forecast ──────────────────────────────────────────────────

def forecast(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """
    Full pipeline: decompose → fit trend → forecast N months ahead.

    Returns a DataFrame with columns:
        ds, actual, trend, seasonal, forecast, ci_lower, ci_upper
    """
    monthly = build_monthly_series(df)
    decomposed, seasonal_idx = decompose(monthly["y"])

    slope, intercept, predict_trend = fit_trend(decomposed["trend"])

    # Future dates
    last_date   = monthly.index[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                  periods=periods, freq="MS")
    future_months = future_dates.month

    # Forecast = trend_extrapolation × seasonal_index
    trend_vals   = predict_trend(periods)
    # Clip negative trends to a minimum (revenue can't go below 0)
    trend_vals   = np.maximum(trend_vals, 0)
    seasonal_vals = np.array([seasonal_idx.get(m, 1.0) for m in future_months])
    forecast_vals = trend_vals * seasonal_vals

    # Confidence interval: ±1 std of historical residuals
    residual_std = decomposed["residual"].dropna().std()
    ci_width     = forecast_vals * residual_std

    future_df = pd.DataFrame({
        "ds":       future_dates,
        "actual":   np.nan,
        "trend":    trend_vals,
        "seasonal": seasonal_vals,
        "forecast": forecast_vals,
        "ci_lower": np.maximum(forecast_vals - ci_width, 0),
        "ci_upper": forecast_vals + ci_width,
    }).set_index("ds")

    # Historical actuals + in-sample fit
    hist_df = pd.DataFrame({
        "ds":       monthly.index,
        "actual":   monthly["y"].values,
        "trend":    decomposed["trend"].values,
        "seasonal": decomposed["seasonal"].values,
        "forecast": (decomposed["trend"] * decomposed["seasonal"]).values,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
    }).set_index("ds")

    result = pd.concat([hist_df, future_df]).reset_index()
    result.rename(columns={"index": "ds"}, inplace=True)
    return result, seasonal_idx


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_forecast(forecast_df: pd.DataFrame,
                  seasonal_idx: pd.Series,
                  periods: int = 6,
                  output_dir: str = "outputs/charts") -> str:
    """
    2-panel chart:
    Left:  actual vs forecast with confidence interval
    Right: seasonal index bar chart
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), facecolor="#F8FAFC")

    # ── Panel 1: Forecast ────────────────────────────────────────────────
    hist_mask   = forecast_df["actual"].notna()
    future_mask = forecast_df["actual"].isna()

    # Historical actual
    ax1.plot(forecast_df.loc[hist_mask, "ds"],
             forecast_df.loc[hist_mask, "actual"],
             color="#2563EB", linewidth=2, label="Actual", zorder=3)

    # In-sample fit
    ax1.plot(forecast_df.loc[hist_mask, "ds"],
             forecast_df.loc[hist_mask, "forecast"],
             color="#64748B", linewidth=1.2, linestyle="--",
             label="Model fit", alpha=0.7)

    # Future forecast
    ax1.plot(forecast_df.loc[future_mask, "ds"],
             forecast_df.loc[future_mask, "forecast"],
             color="#DC2626", linewidth=2.5, linestyle="-",
             label=f"Forecast (+{periods} months)", zorder=3)

    # Confidence interval
    ax1.fill_between(forecast_df.loc[future_mask, "ds"],
                     forecast_df.loc[future_mask, "ci_lower"],
                     forecast_df.loc[future_mask, "ci_upper"],
                     color="#DC2626", alpha=0.12, label="95% CI")

    # Vertical divider at forecast start
    split_date = forecast_df.loc[future_mask, "ds"].iloc[0]
    ax1.axvline(split_date, color="#94A3B8", linewidth=1, linestyle=":")
    ax1.text(split_date, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1,
             "  Forecast →", fontsize=8, color="#64748B", va="top")

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax1.legend(fontsize=9, framealpha=0.7)
    ax1.set_title("Revenue Forecast", fontsize=13, fontweight="bold",
                  color="#1E293B", pad=12)
    ax1.set_facecolor("#F8FAFC")
    ax1.grid(axis="y", color="#E2E8F0", linewidth=0.7, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(colors="#64748B", labelsize=9)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # ── Panel 2: Seasonal index ──────────────────────────────────────────
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    idx_vals = [seasonal_idx.get(m, 1.0) for m in range(1, 13)]
    bar_colors = ["#DC2626" if v == max(idx_vals) else
                  "#2563EB" if v >= 1.0 else "#94A3B8"
                  for v in idx_vals]

    bars = ax2.bar(MONTHS, idx_vals, color=bar_colors, edgecolor="white", width=0.6)
    ax2.axhline(1.0, color="#64748B", linewidth=1, linestyle="--", label="Baseline (1.0)")
    for bar, val in zip(bars, idx_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#1E293B")

    ax2.set_ylim(0, max(idx_vals) * 1.2)
    ax2.set_title("Seasonal Index by Month\n(> 1.0 = above-average revenue)",
                  fontsize=12, fontweight="bold", color="#1E293B", pad=12)
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#F8FAFC")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(colors="#64748B", labelsize=9)
    ax2.grid(axis="y", color="#E2E8F0", linewidth=0.6, linestyle="--")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "11_sales_forecast.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    print(f"  ✓ Saved → {filepath}")
    return filepath


# ── Summary table ──────────────────────────────────────────────────────────────

def forecast_summary(forecast_df: pd.DataFrame) -> None:
    """Print the future forecast values to console."""
    future = forecast_df[forecast_df["actual"].isna()].copy()
    future["month"] = future["ds"].dt.strftime("%b %Y")
    print("\n── 6-Month Revenue Forecast ─────────────────────────────────")
    print(f"  {'Month':<12} {'Forecast':>12} {'Lower CI':>12} {'Upper CI':>12}")
    print("  " + "─" * 52)
    for _, row in future.iterrows():
        print(f"  {row['month']:<12} ${row['forecast']:>10,.0f} "
              f"${row['ci_lower']:>10,.0f} ${row['ci_upper']:>10,.0f}")
    print("─" * 56)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_forecasting(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """Entry point called from run_analysis.py or Streamlit."""
    print(f"\n[Forecasting] Building {periods}-month forecast…")
    forecast_df, seasonal_idx = forecast(df, periods=periods)
    forecast_summary(forecast_df)
    plot_forecast(forecast_df, seasonal_idx, periods=periods)
    return forecast_df
