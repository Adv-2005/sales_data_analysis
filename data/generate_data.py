"""
generate_data.py
----------------
Generates a realistic 12,000-row sales dataset.
Run once: python data/generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────

N_ROWS = 12_000

CATEGORIES = {
    "Technology": ["Laptop", "Monitor", "Keyboard", "Mouse", "Webcam",
                   "Headphones", "USB Hub", "SSD Drive", "Tablet", "Printer"],
    "Furniture":  ["Office Chair", "Standing Desk", "Bookshelf", "Filing Cabinet",
                   "Whiteboard", "Desk Lamp", "Monitor Stand", "Storage Ottoman"],
    "Office Supplies": ["Notebook", "Pen Set", "Stapler", "Paper Ream",
                        "Sticky Notes", "Binder", "Highlighters", "Calendar",
                        "Tape Dispenser", "Scissors"],
}

PRICE_MAP = {
    "Laptop": 1200, "Monitor": 350, "Keyboard": 85, "Mouse": 45,
    "Webcam": 95, "Headphones": 180, "USB Hub": 40, "SSD Drive": 130,
    "Tablet": 500, "Printer": 280,
    "Office Chair": 320, "Standing Desk": 680, "Bookshelf": 220,
    "Filing Cabinet": 180, "Whiteboard": 150, "Desk Lamp": 65,
    "Monitor Stand": 55, "Storage Ottoman": 110,
    "Notebook": 12, "Pen Set": 18, "Stapler": 22, "Paper Ream": 8,
    "Sticky Notes": 6, "Binder": 9, "Highlighters": 14,
    "Calendar": 16, "Tape Dispenser": 11, "Scissors": 10,
}

REGIONS = {
    "West":    ["Los Angeles", "San Francisco", "Seattle", "Phoenix", "Denver"],
    "East":    ["New York", "Boston", "Philadelphia", "Miami", "Atlanta"],
    "Central": ["Chicago", "Dallas", "Houston", "Minneapolis", "Kansas City"],
    "South":   ["New Orleans", "Nashville", "Charlotte", "Tampa", "Austin"],
}

SEGMENTS = ["Consumer", "Corporate", "Home Office"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def random_date(start="2022-01-01", end="2024-12-31"):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days
    # Seasonal weight: boost Nov–Dec by 3x
    date = start_dt + timedelta(days=random.randint(0, delta))
    # Resample ~20% of dates into Q4 for realism
    if random.random() < 0.20:
        year   = random.choice([2022, 2023, 2024])
        month  = random.choice([11, 12])
        day    = random.randint(1, 28)
        date   = datetime(year, month, day)
    return date


def build_row(order_id):
    category = random.choices(
        list(CATEGORIES.keys()), weights=[0.35, 0.25, 0.40]
    )[0]
    product  = random.choice(CATEGORIES[category])
    base_price = PRICE_MAP[product]
    # Add ±15% price noise
    price    = round(base_price * np.random.uniform(0.85, 1.15), 2)
    quantity = int(np.random.choice(range(1, 11), p=[.30,.25,.15,.10,.07,.05,.03,.02,.02,.01]))
    discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20, 0.25]), 2)
    revenue  = round(price * quantity * (1 - discount), 2)
    # Profit margin varies by category
    margin   = {"Technology": 0.18, "Furniture": 0.22, "Office Supplies": 0.35}[category]
    profit   = round(revenue * np.random.uniform(margin - 0.05, margin + 0.05), 2)

    region   = random.choices(list(REGIONS.keys()), weights=[0.30, 0.35, 0.20, 0.15])[0]
    city     = random.choice(REGIONS[region])
    date     = random_date()

    return {
        "order_id":    f"ORD-{order_id:06d}",
        "customer_id": f"CUST-{random.randint(1000, 3999):04d}",
        "segment":     random.choices(SEGMENTS, weights=[0.50, 0.35, 0.15])[0],
        "date":        date.strftime("%Y-%m-%d"),
        "product":     product,
        "category":    category,
        "region":      region,
        "city":        city,
        "quantity":    quantity,
        "unit_price":  price,
        "discount":    discount,
        "revenue":     revenue,
        "profit":      profit,
    }


# ── Inject dirt (for cleaning demo) ─────────────────────────────────────────

def inject_noise(df):
    df_noisy = df.copy()
    idx = df_noisy.index.tolist()

    # 1. Missing values (~2%)
    for col in ["discount", "city", "segment"]:
        drop_idx = np.random.choice(idx, size=int(N_ROWS * 0.02), replace=False)
        df_noisy.loc[drop_idx, col] = np.nan

    # 2. Duplicate rows (~1%)
    dup_rows = df_noisy.sample(frac=0.01, random_state=1)
    df_noisy = pd.concat([df_noisy, dup_rows], ignore_index=True)

    # 3. Column name inconsistency
    df_noisy.rename(columns={"revenue": "Revenue ", "profit": "Profit"},
                    inplace=True)

    # 4. Mixed-case category
    bad_idx = np.random.choice(df_noisy.index, size=200, replace=False)
    df_noisy.loc[bad_idx, "category"] = df_noisy.loc[bad_idx, "category"].str.lower()

    # 5. Negative quantity outlier
    df_noisy.loc[np.random.choice(df_noisy.index, 20), "quantity"] = -1

    return df_noisy


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating dataset…")
    rows   = [build_row(i + 1) for i in range(N_ROWS)]
    df     = pd.DataFrame(rows)
    dirty  = inject_noise(df)
    output = "data/sales_raw.csv"
    dirty.to_csv(output, index=False)
    print(f"✓ Saved {len(dirty):,} rows → {output}")
    print(dirty.head(3))
