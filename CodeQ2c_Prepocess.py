import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ------------------ Step 1: Load Dataset ------------------
file_path = "dataset_for_final.csv"  # Original Dataset
df = pd.read_csv(file_path)

# ------------------ Step 2: Define Lookup Mappings ------------------
job_status_mapping = {
    "-2": "Not applicable",
    "1": "Permanent",
    "2": "Contract",
    "3": "Temporary",
    "4": "Part-time",
    "5": "Working for family"
}

job_assurance_mapping = {
    4: "Self-employed/Freelance",
    6: "Employer",
    7: "Government Employees",
    8: "Private workers (including NGOs)",
    9: "Workers (government/private/family/workers with wages/salaries)",
    40: "Free",
    44: "Self-employment (SKPG 2.0)",
    51: "Working with family (wage/salary)",
    52: "Work with family (no wages/salary)"
}

# ------------------ Step 3: Apply Mappings ------------------
df["JobStatusCategory"] = df["JobStatusCategory"].astype(str).map(job_status_mapping)
df["jobassurance"] = df["jobassurance"].map(job_assurance_mapping)

# ------------------ Step 4: Convert Job Income Range to Numerical ------------------
def convert_income(income):
    try:
        if pd.isna(income) or not isinstance(income, str):
            return np.nan

        income = income.replace("RM", "").replace(",", "")

        if "-" in income:
            low, high = income.split("-")
            return (int(low.strip()) + int(high.strip())) / 2
        elif "Bawah" in income:
            return int(''.join(filter(str.isdigit, income))) - 1
        elif "Lebih" in income:
            return int(''.join(filter(str.isdigit, income))) + 1
        elif income.isdigit():
            return int(income)
    except:
        return np.nan
    return np.nan

df["JobIncome"] = df["JobIncome"].astype(str).apply(convert_income)

# ------------------ Step 5: Categorize Job Income ------------------
income_bins = [0, 3000, 7000, np.inf]  # Adjust thresholds as needed
income_labels = [0, 1, 2]  # Lower: 0, Moderate: 1, Higher: 2
df["JobIncomeCategory"] = pd.cut(df["JobIncome"], bins=income_bins, labels=income_labels)

# Fix: Avoid FutureWarning for filling missing values
df["JobIncomeCategory"] = df["JobIncomeCategory"].fillna(df["JobIncomeCategory"].mode()[0])

# Drop original JobIncome column as it's now categorized
df.drop(columns=["JobIncome"], inplace=True)

# ------------------ Step 6: Handle Missing Values ------------------
categorical_cols = ["JobStatusCategory", "jobassurance", "gender", "Nationality", "EntryChannel"]
numerical_cols = ["AgeGrad", "CGPA"]

# Fix: Avoid FutureWarning by assigning back to DataFrame
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# ------------------ Step 7: Encode Categorical Variables ------------------
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for possible decoding

# ------------------ Step 8: Normalize Numerical Features ------------------
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ------------------ Step 9: Save Preprocessed Data ------------------
output_file = "InputQ2c_Cluster.csv"
df.to_csv(output_file, index=False)
print(f"Preprocessing complete. File saved at: {output_file}")
