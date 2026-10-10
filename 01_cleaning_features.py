"""Step 1: load the raw orders, clean them and add a few useful columns.
The result is saved as data/orders_clean.csv.
"""
import pandas as pd
import numpy as np

orders = pd.read_csv("data/orders.csv", encoding="latin1")
print("Rows loaded:", len(orders))

# Standardise column names
orders.columns = (orders.columns.str.strip().str.lower()
                  .str.replace(" ", "_"))

# Parse dates
for col in ["order_date", "ship_date", "delivery_date"]:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Remove duplicates and impossible records
orders = orders.drop_duplicates(subset="order_id")
orders = orders.dropna(subset=["order_date", "delivery_date"])
orders = orders[orders["delivery_date"] >= orders["order_date"]]

# Handle missing values
orders["shipping_mode"] = orders["shipping_mode"].fillna("Unknown")
orders["distance_km"] = orders.groupby("region")["distance_km"].transform(
    lambda s: s.fillna(s.median()))

# Treat extreme shipping costs as outliers using the IQR rule
q1, q3 = orders["shipping_cost"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
orders["cost_outlier"] = orders["shipping_cost"] > upper

# Feature engineering
orders["lead_time_days"] = (orders["delivery_date"] - orders["order_date"]).dt.days
orders["late"] = (orders["lead_time_days"] > orders["promised_days"]).astype(int)
orders["order_month"] = orders["order_date"].dt.to_period("M").astype(str)
orders["is_weekend"] = orders["order_date"].dt.dayofweek >= 5

orders.to_csv("data/orders_clean.csv", index=False)
print("Rows after cleaning:", len(orders))
print("Saved data/orders_clean.csv")
