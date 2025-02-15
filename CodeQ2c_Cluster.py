import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ------------------ Step 1: Load Preprocessed Data ------------------
file_path = "InputQ2c_Cluster.csv"
df = pd.read_csv(file_path)

# Print column names to verify correct selection
print("Dataset Columns:", df.columns)

# ------------------ Step 2: Ensure All Data is Numeric ------------------
# Check for non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns
if len(non_numeric_columns) > 0:
    print("Non-numeric columns detected:", non_numeric_columns)

    # Drop non-numeric columns (if any)
    df = df.drop(columns=non_numeric_columns)
    print("Dropped non-numeric columns to ensure clustering works correctly.")

# ------------------ Step 3: Elbow Method to Find Optimal K ------------------
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method to Determine Optimal k")
plt.show()

# ------------------ Step 4: Apply K-Means Clustering ------------------
optimal_k = 3  # Adjust based on elbow method results
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df)

# ------------------ Step 5: Visualize Clusters ------------------
x_feature = "AgeGrad"
y_feature = "CGPA"

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df[x_feature], y=df[y_feature], hue=df['Cluster'],
                palette=['blue', 'green', 'purple'])
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title(f"K-Means Clustering (k={optimal_k})")
plt.legend(title="Cluster")
plt.show()

# ------------------ Step 6: Save Clustered Data ------------------
output_file = "Clustered_Data.csv"
df.to_csv(output_file, index=False)
print(f"Clustering complete. File saved at: {output_file}")
