import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ------------------ Step 1: Load Dataset ------------------
file_path = "dataset_for_final.csv"  # Original dataset
df = pd.read_csv(file_path)

# ------------------ Step 2: Handle Missing Values ------------------
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].mean())  # Assign the filled values back
    else:
        df[col] = df[col].fillna(df[col].mode()[0])  # Assign the filled values back

# ------------------ Step 3: Encode Categorical Variables ------------------
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for future decoding

# ------------------ Step 4: Normalize Numerical Features ------------------
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# ------------------ Step 5: Save Preprocessed Data ------------------
df_scaled.to_csv("InputQ2c_Cluster.csv", index=False)
print("Preprocessed data saved as InputQ2c_Cluster.csv")
