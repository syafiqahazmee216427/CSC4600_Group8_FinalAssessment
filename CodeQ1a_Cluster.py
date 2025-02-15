import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("InputQ1a_Cluster.csv")

# Select Weight and MRP as the features for clustering
col_x = "Weight"
col_y = "MRP"

# Ensure the selected features exist in the dataset
if col_x not in df.columns or col_y not in df.columns:
    raise ValueError(f"Columns '{col_x}' and '{col_y}' not found in dataset.")

# Extract only the required features
df_selected = df[[col_x, col_y]]

# Standardize the data (to improve clustering performance)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)

# Try different k values and select the best one based on silhouette score
k_values = [3, 5]
best_k = None
best_silhouette = -1

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    silhouette = silhouette_score(df_scaled, clusters)

    print(f"\nFor k = {k}:")
    print(f"- Inertia (SSE): {kmeans.inertia_:.2f}")
    print(f"- Silhouette Score: {silhouette:.4f}")

    if silhouette > best_silhouette:
        best_k = k
        best_silhouette = silhouette

# Apply k-Means with the best k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster_Label"] = kmeans.fit_predict(df_scaled)

# Scatter plot for visualization
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df[col_x], df[col_y], c=df["Cluster_Label"], cmap='viridis', alpha=0.6, label="Data Points")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label="Centroids")

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label(f"Cluster Number (k={best_k})")

# Plot settings with actual column names
plt.title(f'k-Means Clustering with k={best_k}')
plt.xlabel(col_x)
plt.ylabel(col_y)
plt.legend()
plt.show()

# Save clustered data for classification
df.to_csv("Clustered_Data.csv", index=False)
print(f"\nClustered data saved as 'Clustered_Data.csv' with k = {best_k}")
