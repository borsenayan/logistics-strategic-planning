# Logistics Strategic Planning and Data Exploration

This is the code for my Week 1 task in the Yuva Internship. In the report I planned a data analysis project for RegionalKart, a made up ecommerce company in Maharashtra. This repository has the Python code that shows how I would carry out that plan.

## The problems I looked at
1. Late deliveries
2. Stockouts and overstock
3. High cost per delivery

## What the code does
It cleans order data, calculates KPIs, explores the data with charts, predicts delivery lead time, groups delivery zones with KMeans, forecasts demand, sets a reorder point and plans vehicle routes.

## Folder layout
```
src/
  generate_sample_data.py     makes up orders and inventory data
  01_cleaning_features.py     cleaning and new columns
  02_kpis_eda.py              baseline KPIs and charts
  03_lead_time_regression.py  lead time regression and late order classifier
  04_zone_clustering.py       KMeans clusters of delivery zones
  05_forecast_inventory.py    Holt Winters forecast, EOQ and reorder point
  06_route_optimization.py    vehicle routing with OR-Tools
data/       generated CSV files (not tracked by Git)
outputs/    charts and result tables
```

## How to run it
```
pip install -r requirements.txt
python src/generate_sample_data.py
python src/01_cleaning_features.py
python src/02_kpis_eda.py
python src/03_lead_time_regression.py
python src/04_zone_clustering.py
python src/05_forecast_inventory.py
python src/06_route_optimization.py
```
Run every command from the main project folder.

## KPIs I track
On Time Delivery Rate, Average Order Lead Time, Cost per Delivery, Inventory Turnover, Stockout Rate, Demand Forecast Accuracy and Vehicle Capacity Utilisation.

## A note about the data
The data here is made up. It is only meant to show how the method works and the numbers don't describe a real business. If you want to use a public dataset such as DataCo Smart Supply Chain or the Olist Brazilian E-Commerce dataset, rename its columns to match `generate_sample_data.py` and save it as `data/orders.csv`.

## About me
Nayan Deepak Borse, B.E. Computer Engineering (2026)
