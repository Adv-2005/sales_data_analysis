# 📊 Sales Data Analysis Pipeline

A production-quality Python data analysis project demonstrating end-to-end skills:
data cleaning → EDA → visualisation → business insights.

Built as a portfolio project for Data Analyst / Data Scientist roles.

---

## 🗂️ Project Structure

```
sales-analysis/
├── data/
│   ├── generate_data.py     # Synthetic dataset generator (12k rows)
│   └── sales_clean.csv      # Output of cleaning pipeline
├── src/
│   ├── data_cleaning.py     # Modular cleaning & feature engineering
│   ├── eda.py               # All EDA functions (returns DataFrames)
│   └── visualization.py     # 8 Matplotlib charts
├── outputs/
│   ├── charts/              # PNG exports (8 charts)
│   └── report.txt           # Auto-generated insights report
├── run_analysis.py          # Master pipeline script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/your-username/sales-analysis.git
cd sales-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python data/generate_data.py

# 4. Run the full pipeline
python run_analysis.py
```

**Total execution time: ~2–3 seconds on a standard laptop.**

---

## 📦 Dataset

- **12,120 rows** (after cleaning: ~11,983)
- **3 years** of synthetic sales data (2022–2024)
- **Columns**: Order ID, Customer ID, Segment, Date, Product, Category,
  Region, City, Quantity, Unit Price, Discount, Revenue, Profit

### Intentional data quality issues (cleaned in pipeline):
| Issue | Count | Fix Applied |
|---|---|---|
| Missing values | ~2% per column | Column-targeted fill strategy |
| Duplicate rows | 117 | `drop_duplicates()` |
| Inconsistent column names | `"Revenue "` with trailing space | `str.strip().str.lower()` |
| Mixed-case categories | 200 rows | `.str.title()` |
| Negative quantities | 20 rows | Business-rule filter |

---

## 🧹 Data Cleaning (`src/data_cleaning.py`)

Each step is a pure function — testable and reusable independently.

```
load_data()            → Read CSV, print shape
standardize_columns()  → Strip / lowercase / snake_case all column names
fix_dtypes()           → Parse dates, cast numerics
handle_missing()       → Targeted per-column fill strategy
remove_duplicates()    → Keep first occurrence
fix_quality_issues()   → Business-rule validation + category normalisation
engineer_features()    → Add year / month / quarter / year_month / profit_margin
```

**Key decision**: Revenue is *recomputed* from `unit_price × quantity × (1 − discount)`
after cleaning to guarantee a consistent source of truth downstream.

---

## 📈 Exploratory Data Analysis (`src/eda.py`)

| Function | Answers |
|---|---|
| `summary_stats()` | Total revenue, profit, orders, unique customers, AOV |
| `monthly_sales_trend()` | Sales trend over time |
| `top_products()` | Top N products by revenue + margin |
| `category_performance()` | Revenue share + avg margin per category |
| `region_performance()` | Revenue, profit, orders, customer count per region |
| `top_customers()` | VIP customer identification |
| `segment_analysis()` | Consumer vs Corporate vs Home Office |
| `monthly_avg_revenue()` | Seasonal pattern (monthly average across years) |
| `heatmap_data()` | Month × Region revenue pivot |
| `discount_impact()` | Margin erosion per discount tier |

---

## 📊 Visualisations (`outputs/charts/`)

All built with **pure Matplotlib** (no Seaborn).

| # | Chart | Type | Key Insight |
|---|---|---|---|
| 01 | Monthly Revenue & Profit Trend | Dual-axis line | Seasonal spikes visible |
| 02 | Top 10 Products by Revenue | Horizontal bar | Laptop dominates at $1.36M |
| 03 | Category Overview | Pie + bar | Office Supplies has 34.9% margin |
| 04 | Region Performance | Grouped bar | East leads; South underperforms |
| 05 | Order Value Distribution | Histogram | Right-skewed; mean > median |
| 06 | Monthly Seasonality | Bar | November is peak month |
| 07 | Revenue Heatmap (Month × Region) | Custom heatmap | Q4 East dominates |
| 08 | Top 10 Customers | Horizontal bar | Customer concentration ~2.7% |

---

## 💡 Business Insights

Extracted programmatically — see full report at `outputs/report.txt`.

1. **Peak Month is November** → Increase inventory & marketing 4 weeks prior
2. **East region leads** ($1.75M) vs **South lags** ($698K) → Investigate South's logistics
3. **Office Supplies has the best margin** (34.9%) → Prioritise in upsell campaigns
4. **Laptop = top product** ($1.36M, 419 orders) → Never let it go out of stock
5. **Heavy discounts (21–30%) erode margins** → Cap standard discounts at 10%
6. **Top 10 customers = 2.7% of revenue** → Build a VIP retention programme
7. **Revenue declined 4.4% YoY (2023→2024)** → Audit churn + pricing strategy

---

## 🔭 Future Enhancements

Mention these in interviews to show forward thinking:

- **Streamlit dashboard** — interactive version of all charts
- **Time-series forecasting** — Prophet / ARIMA for demand planning
- **RFM segmentation** — Recency, Frequency, Monetary customer scoring
- **ML churn model** — XGBoost classifier on customer behaviour
- **Automated alerts** — Flag regional revenue dips > 15% MoM

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Pandas | 2.x | Data manipulation |
| NumPy | 1.24+ | Numerical ops |
| Matplotlib | 3.7+ | All visualisations |

---

## 🎯 Skills Demonstrated

✅ Data wrangling at scale (12k+ rows)  
✅ Modular, production-style code architecture  
✅ Business-first EDA (not just plots)  
✅ Custom Matplotlib visualisations  
✅ Automated insight generation  
✅ Written communication (report.txt)

---

*Built as a portfolio project. Dataset is synthetic but designed to mirror
real retail sales patterns including seasonality, regional variation, and
discount behaviour.*
