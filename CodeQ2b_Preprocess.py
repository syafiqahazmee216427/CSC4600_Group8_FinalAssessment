import pandas as pd

# Step 1: Load dataset
file_path = "dataset_for_final.csv"
df = pd.read_csv(file_path)

# Step 2: Define lookup tables
job_status_mapping = {
    -2: "Not applicable",
    1: "Permanent",
    2: "Contract",
    3: "Temporary",
    4: "Part-time",
    5: "Working for family"
}

job_assurance_mapping = {
    4: "Self-employed/Freelance",
    6: "Employer",
    7: "Government employees",
    8: "Private workers (including NGOs)",
    9: "Workers (government/private/family wages)",
    40: "Free",
    44: "Self-employment (SKPG 2.0)",
    51: "Working with family (wage/salary)",
    52: "Work with family (no wages/salary)"
}

# Step 3: Convert to numeric to prevent mapping issues
df["JobStatusCategory"] = pd.to_numeric(df["JobStatusCategory"], errors='coerce')
df["jobassurance"] = pd.to_numeric(df["jobassurance"], errors='coerce')

# Step 4: Apply lookup tables using .replace()
df["JobStatusCategory"] = df["JobStatusCategory"].replace(job_status_mapping)
df["jobassurance"] = df["jobassurance"].replace(job_assurance_mapping)

# Step 5: Drop NaN values if any
df.dropna(subset=["jobassurance", "JobStatusCategory"], inplace=True)

# Step 6: Save preprocessed file
df.to_csv("InputQ2b_Association.csv", index=False)
print("Preprocessing complete. File saved as 'InputQ2b_Association.csv'.")
