import pandas as pd

# Step 1: Load the dataset
file_path = "dataset_for_final.csv"  # original dataset
df = pd.read_csv(file_path)

# Step 2: Define lookup tables for categorical mapping
job_status_lookup = {
    -2: "Not applicable",
    1: "Permanent",
    2: "Contract",
    3: "Temporary",
    4: "Part time",
    5: "Working for family"
}

job_assurance_lookup = {
    4: "Self-employed/Freelance",
    6: "Employer",
    7: "Government employees",
    8: "Private workers (including NGOs)",
    9: "Workers (government/private/family workers with wages/salaries)",
    40: "Free",
    44: "Self-employment (SKPG 2.0)",
    51: "Working with family (wage/salary)",
    52: "Work with family (no wages/salary)"
}

# Step 3: Apply lookup tables
df["JobStatusCategory"] = df["JobStatusCategory"].map(job_status_lookup).fillna("Unknown")
df["jobassurance"] = df["jobassurance"].map(job_assurance_lookup).fillna("Unknown")

# Step 4: Save preprocessed file
df.to_csv("InputQ2b_Association.csv", index=False)

print("Preprocessing complete. File saved as 'InputQ2b_Association.csv'.")
