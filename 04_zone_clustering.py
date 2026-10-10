"""Step 4: group the delivery zones into clusters with KMeans.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

orders = pd.read_csv("data/orders_clean.csv")

zones = orders.groupby("delivery_zone").agg(
    avg_lead=("lead_time_days", "mean"),
    late_rate=("late", "mean"),
    volume=("order_id", "count"),
    avg_distance=("distance_km", "mean"))

Z = StandardScaler().fit_transform(zones)

# Choose the number of clusters with the silhouette score
scores = {}
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
    scores[k] = silhouette_score(Z, labels)

best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()})
print("Best number of clusters:", best_k)

zones["cluster"] = KMeans(n_clusters=best_k, n_init=10,
                          random_state=42).fit_predict(Z)

# Profile each cluster so that managers can name and act on it
print(zones.groupby("cluster").mean().round(2))
zones.to_csv("outputs/zone_clusters.csv")
