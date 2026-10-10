"""Step 5: forecast monthly demand and work out a simple stock policy
(economic order quantity and reorder point).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

orders = pd.read_csv("data/orders_clean.csv", parse_dates=["order_date"])

monthly_demand = (orders.set_index("order_date")
                  .resample("MS")["order_quantity"].sum())

train = monthly_demand[:-6]
test = monthly_demand[-6:]

fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                           seasonal_periods=12).fit()
forecast = fit.forecast(6)

mape = np.mean(np.abs((test - forecast) / test)) * 100
print("Forecast MAPE:", round(mape, 2), "%")
print("Forecast accuracy:", round(100 - mape, 2), "%")


# Inventory policy: economic order quantity and reorder point
def eoq(annual_demand, order_cost, holding_cost):
    return np.sqrt(2 * annual_demand * order_cost / holding_cost)


def reorder_point(daily_demand, daily_std, lead_time, service_level=0.95):
    z = norm.ppf(service_level)
    safety_stock = z * daily_std * np.sqrt(lead_time)
    return daily_demand * lead_time + safety_stock


daily = orders.groupby("order_date")["order_quantity"].sum()
annual = daily.mean() * 365
print("\nExample policy for the whole network")
print("EOQ (units):", round(eoq(annual, order_cost=2500, holding_cost=40)))
print("Reorder point (units):",
      round(reorder_point(daily.mean(), daily.std(), lead_time=3)))
