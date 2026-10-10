"""Step 2: work out the baseline KPIs and make a couple of charts.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

orders = pd.read_csv("data/orders_clean.csv", parse_dates=["order_date"])
inventory = pd.read_csv("data/inventory.csv", parse_dates=["date"])

# Headline KPIs
otd_rate = (1 - orders["late"].mean()) * 100
avg_lead_time = orders["lead_time_days"].mean()
cost_per_delivery = orders["shipping_cost"].sum() / len(orders)

# Monthly KPI table that can feed a Power BI dashboard
monthly = orders.groupby("order_month").agg(
    orders=("order_id", "count"),
    otd_pct=("late", lambda s: (1 - s.mean()) * 100),
    avg_lead=("lead_time_days", "mean"),
    total_cost=("shipping_cost", "sum"))
monthly.to_csv("outputs/monthly_kpis.csv")

# Inventory KPIs
years = (inventory["date"].max() - inventory["date"].min()).days / 365
annual_cogs = inventory["cogs"].sum() / years
avg_inventory_value = inventory.groupby("date")["stock_value"].sum().mean()
inventory_turnover = annual_cogs / avg_inventory_value
stockout_rate = (inventory["closing_stock"] == 0).mean() * 100

print(f"On time delivery rate : {otd_rate:.1f}%")
print(f"Average lead time     : {avg_lead_time:.2f} days")
print(f"Cost per delivery     : Rs {cost_per_delivery:.2f}")
print(f"Inventory turnover    : {inventory_turnover:.2f}")
print(f"Stockout rate         : {stockout_rate:.2f}%")

# Exploratory views
plt.figure(figsize=(7, 4))
sns.boxplot(data=orders, x="shipping_mode", y="lead_time_days")
plt.title("Lead time by shipping mode")
plt.tight_layout()
plt.savefig("outputs/lead_time_by_mode.png", dpi=150)
plt.close()

region_late = (orders.groupby("region")["late"].mean()
               .sort_values(ascending=False) * 100)
print("\nLate delivery percentage by region")
print(region_late.round(1))

plt.figure(figsize=(6, 5))
sns.heatmap(orders[["lead_time_days", "distance_km", "order_quantity",
                    "shipping_cost"]].corr(), annot=True, cmap="Blues")
plt.title("Correlation of key numeric fields")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()
print("\nSaved charts and monthly_kpis.csv in outputs/")
