import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_association = pd.read_csv("InputQ2b_Association.csv")  # For insights 1, 2, and 3
df_cluster = pd.read_csv("InputQ2c_Cluster.csv")  # For insights 4 and 5

# Insight 1: Job Status Category vs. Job Assurance (Heatmap)
plt.figure(figsize=(8, 5))
job_status_pivot = df_association.pivot_table(index='JobStatusCategory', columns='jobassurance', aggfunc='size', fill_value=0)
sns.heatmap(job_status_pivot, annot=True, cmap='Blues', fmt='d')
plt.title("Job Status Category vs. Job Assurance")
plt.xlabel("Job Assurance Level")
plt.ylabel("Job Status Category")
plt.xticks(fontsize=5)
plt.show()

# Insight 2: Job Income Distribution (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(df_association['JobIncome'], bins=20, kde=True, color='blue')
plt.title("Distribution of Job Income")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.xticks(fontsize=5)
plt.show()

# Insight 4: Clustering of Graduates by CGPA and Age at Graduation (Scatterplot)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_cluster['CGPA'], y=df_cluster['AgeGrad'], hue=df_cluster['Status'], palette='viridis')
plt.title("Clustering of Graduates by CGPA and Age at Graduation")
plt.xlabel("CGPA")
plt.ylabel("Age at Graduation")
plt.show()

# Insight 5: Job Stability Based on CGPA and Age at Graduation (Line Graph)
plt.figure(figsize=(8, 5))
sns.lineplot(x=df_cluster['AgeGrad'], y=df_cluster['CGPA'], hue=df_cluster['JobStatusCategory'], marker='o')
plt.title("Job Stability by CGPA and Age at Graduation")
plt.xlabel("Age at Graduation")
plt.ylabel("CGPA")
plt.legend(title="Job Stability Category")
plt.show()
