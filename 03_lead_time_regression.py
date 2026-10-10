"""Step 3: predict how many days an order will take and whether it will be late.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report)

orders = pd.read_csv("data/orders_clean.csv")

num_cols = ["distance_km", "order_quantity", "promised_days"]
cat_cols = ["shipping_mode", "region", "product_category"]
X = orders[num_cols + cat_cols]
y = orders["lead_time_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42,
                                           n_jobs=-1),
}

for name, reg in models.items():
    pipe = Pipeline([("prep", prep), ("model", reg)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"{name}: MAE = {mean_absolute_error(y_test, pred):.3f} days, "
          f"R2 = {r2_score(y_test, pred):.3f}")

# Classification version: will this order be late?
y_late = orders["late"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_late, test_size=0.2, random_state=42, stratify=y_late)
clf = Pipeline([("prep", prep),
                ("model", RandomForestClassifier(n_estimators=100,
                                                 random_state=42, n_jobs=-1))])
clf.fit(Xc_train, yc_train)
print("\nLate delivery risk model")
print(classification_report(yc_test, clf.predict(Xc_test)))
