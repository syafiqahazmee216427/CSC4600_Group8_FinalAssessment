import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ------------------ Step 1: Load Preprocessed Data ------------------
file_path = "InputQ2c_Cluster.csv"
df_scaled = pd.read_csv(file_path)

# Print column names to verify correct selection
print(df_scaled.columns)

# ------------------ Step 2: Elbow Method to Find Optimal K ------------------
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method to Determine Optimal k")
plt.show()

# ------------------ Step 3: Apply K-Means Clustering ------------------
optimal_k = 3  # This is based on the elbow method result previously
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)

# ------------------ Step 4: Visualize Clusters ------------------
# Corrected feature names based on column list
x_feature = "AgeGrad"  # Corrected column name
y_feature = "CGPA"  # CGPA is correct

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_scaled[x_feature], y=df_scaled[y_feature], hue=df_scaled['Cluster'],
                palette=['blue', 'green', 'purple'])
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title(f"K-Means Clustering (k={optimal_k})")
plt.legend(title="Cluster")
plt.show()
