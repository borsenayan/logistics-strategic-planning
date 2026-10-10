"""I don't have real company data, so this script makes up a dataset for a
fictional ecommerce company in Maharashtra. It is only here so the other
scripts have something to run on.

To try a public dataset such as DataCo or Olist, rename its columns to match
the ones below and save it as data/orders.csv.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_ORDERS = 30000
START = pd.Timestamp("2023-01-01")
N_DAYS = 1095

regions = ["Mumbai", "Pune", "Nashik", "Nagpur", "Sambhajinagar"]
categories = ["Fashion", "Electronics", "Home", "Essentials"]
modes = {"Standard": (3.5, 5), "Express": (1.8, 3), "Same Day": (0.8, 1)}

# More orders in the festive months (October and November)
days = pd.date_range(START, periods=N_DAYS, freq="D")
weights = np.where(days.month.isin([10, 11]), 1.7, 1.0)
weights = weights / weights.sum()
order_dates = rng.choice(days, size=N_ORDERS, p=weights)

region = rng.choice(regions, size=N_ORDERS, p=[0.30, 0.25, 0.15, 0.15, 0.15])
zone = np.array([f"{r}_Z{rng.integers(1, 5)}" for r in region])
mode = rng.choice(list(modes), size=N_ORDERS, p=[0.6, 0.3, 0.1])
category = rng.choice(categories, size=N_ORDERS)
quantity = rng.integers(1, 6, size=N_ORDERS)
distance = rng.gamma(shape=2.0, scale=45.0, size=N_ORDERS).round(1) + 3

base = np.array([modes[m][0] for m in mode])
promised = np.array([modes[m][1] for m in mode])
festive = pd.DatetimeIndex(order_dates).month.isin([10, 11]).astype(int)
lead = base + distance / 120 + festive * 0.8 + rng.normal(0, 0.9, N_ORDERS)
lead = np.clip(np.round(lead), 0, None).astype(int)

mode_factor = np.array([{"Standard": 1.0, "Express": 1.6, "Same Day": 2.4}[m] for m in mode])
cost = (35 + 0.9 * distance * mode_factor + 6 * quantity
        + rng.normal(0, 8, N_ORDERS)).round(2)

orders = pd.DataFrame({
    "order_id": np.arange(100000, 100000 + N_ORDERS),
    "order_date": pd.to_datetime(order_dates),
    "promised_days": promised,
    "shipping_mode": mode,
    "region": region,
    "delivery_zone": zone,
    "product_category": category,
    "order_quantity": quantity,
    "distance_km": distance,
    "shipping_cost": cost,
})
orders["ship_date"] = orders["order_date"] + pd.to_timedelta(np.minimum(lead, 1), unit="D")
orders["delivery_date"] = orders["order_date"] + pd.to_timedelta(lead, unit="D")

# Add a little mess so the cleaning step has real work to do
missing_idx = rng.choice(orders.index, size=300, replace=False)
orders.loc[missing_idx, "distance_km"] = np.nan
dupes = orders.sample(120, random_state=1)
orders = pd.concat([orders, dupes], ignore_index=True)

orders.to_csv("data/orders.csv", index=False)

# Daily inventory by region and product category
rows = []
for r in regions:
    for c in categories:
        demand = rng.poisson(12, size=N_DAYS)
        stock = rng.integers(0, 60, size=N_DAYS)
        price = rng.integers(300, 3000)
        rows.append(pd.DataFrame({
            "date": days,
            "region": r,
            "product_category": c,
            "closing_stock": stock,
            "stock_value": stock * price,
            "cogs": demand * price * 0.7,
        }))
pd.concat(rows, ignore_index=True).to_csv("data/inventory.csv", index=False)
print("Created data/orders.csv and data/inventory.csv")
