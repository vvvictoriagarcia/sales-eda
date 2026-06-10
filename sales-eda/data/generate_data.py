"""
Generates a realistic synthetic e-commerce sales dataset.
Run once: python data/generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────

N_ORDERS = 5000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHTS = [0.20, 0.18, 0.25, 0.22, 0.15]

CATEGORIES = {
    "Electronics":    ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch", "Monitor"],
    "Office":         ["Desk Chair", "Standing Desk", "Printer", "Keyboard", "Webcam", "Notebook"],
    "Home & Garden":  ["Coffee Maker", "Air Purifier", "Blender", "Vacuum", "Desk Lamp", "Plant Pot"],
    "Clothing":       ["Running Shoes", "Hoodie", "Jeans", "T-Shirt", "Jacket", "Backpack"],
    "Sports":         ["Yoga Mat", "Dumbbells", "Resistance Bands", "Water Bottle", "Jump Rope", "Foam Roller"],
}

CATEGORY_WEIGHTS = [0.30, 0.20, 0.18, 0.17, 0.15]

BASE_PRICES = {
    "Laptop": 1200, "Smartphone": 850, "Tablet": 550, "Headphones": 180,
    "Smartwatch": 320, "Monitor": 400, "Desk Chair": 350, "Standing Desk": 600,
    "Printer": 250, "Keyboard": 90, "Webcam": 75, "Notebook": 15,
    "Coffee Maker": 120, "Air Purifier": 200, "Blender": 80, "Vacuum": 300,
    "Desk Lamp": 45, "Plant Pot": 25, "Running Shoes": 130, "Hoodie": 60,
    "Jeans": 75, "T-Shirt": 25, "Jacket": 110, "Backpack": 85,
    "Yoga Mat": 40, "Dumbbells": 65, "Resistance Bands": 20, "Water Bottle": 30,
    "Jump Rope": 15, "Foam Roller": 35,
}

SALESPERSONS = ["Alice M.", "Bob T.", "Clara R.", "David L.", "Eva S.",
                "Frank O.", "Grace K.", "Henry P."]

CUSTOMER_SEGMENTS = ["B2C", "B2B", "Enterprise"]
SEGMENT_WEIGHTS = [0.60, 0.30, 0.10]

# ── Helpers ───────────────────────────────────────────────────────────────────

def random_date(start, end):
    """Random date with seasonality: Q4 and summer get a boost."""
    days = (end - start).days
    dates = []
    weights = []
    for i in range(days):
        d = start + timedelta(days=i)
        w = 1.0
        if d.month in (11, 12):   w = 2.2   # Black Friday / holiday season
        elif d.month in (6, 7, 8): w = 1.4  # Summer boost
        elif d.month in (1, 2):    w = 0.7  # Post-holiday slowdown
        dates.append(d)
        weights.append(w)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return np.random.choice(dates, p=weights)


def build_product_list():
    rows = []
    pid = 1
    for cat, products in CATEGORIES.items():
        for p in products:
            rows.append({"product_id": f"P{pid:03d}", "product_name": p,
                         "category": cat, "base_price": BASE_PRICES[p]})
            pid += 1
    return pd.DataFrame(rows)


# ── Generate orders ───────────────────────────────────────────────────────────

products_df = build_product_list()
category_list = list(CATEGORIES.keys())

records = []
customer_ids = [f"C{i:05d}" for i in range(1, 2001)]  # 2000 unique customers

for i in range(N_ORDERS):
    order_date = random_date(START_DATE, END_DATE)

    # Category → product
    cat = np.random.choice(category_list, p=CATEGORY_WEIGHTS)
    product = products_df[products_df["category"] == cat].sample(1).iloc[0]

    # Pricing
    base = product["base_price"]
    price = round(base * np.random.uniform(0.90, 1.10), 2)
    quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.10, 0.06, 0.04])
    discount = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], p=[0.55, 0.20, 0.13, 0.07, 0.05])
    revenue = round(price * quantity * (1 - discount), 2)
    profit_margin = np.random.uniform(0.08, 0.35)
    profit = round(revenue * profit_margin, 2)

    records.append({
        "order_id":        f"ORD{i+1:05d}",
        "order_date":      order_date.strftime("%Y-%m-%d"),
        "customer_id":     np.random.choice(customer_ids),
        "segment":         np.random.choice(CUSTOMER_SEGMENTS, p=SEGMENT_WEIGHTS),
        "region":          np.random.choice(REGIONS, p=REGION_WEIGHTS),
        "salesperson":     np.random.choice(SALESPERSONS),
        "product_id":      product["product_id"],
        "product_name":    product["product_name"],
        "category":        cat,
        "quantity":        quantity,
        "unit_price":      price,
        "discount":        discount,
        "revenue":         revenue,
        "profit":          profit,
    })

df = pd.DataFrame(records).sort_values("order_date").reset_index(drop=True)

out_path = os.path.join(os.path.dirname(__file__), "raw", "sales_data.csv")
df.to_csv(out_path, index=False)
print(f"Dataset saved: {out_path}")
print(f"Shape: {df.shape}")
print(df.head(3).to_string())
