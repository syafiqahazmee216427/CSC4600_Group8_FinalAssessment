import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Load preprocessed dataset
file_path = "InputQ2b_Association.csv"
df = pd.read_csv(file_path)

# Step 2: Select relevant columns for association rule mining
columns_for_association = ["JobStatusCategory", "jobassurance", "JobIncome"]
df_association = df[columns_for_association].astype(str)

# Step 3: Convert dataframe into a list of transactions
transactions = df_association.fillna("Missing").values.tolist()

# Step 4: Encode transactions into binary format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Step 5: Apply Apriori algorithm with minSup = 0.02 (2%)
minSup = 0.02
frequent_itemsets = apriori(df_encoded, min_support=minSup, use_colnames=True)

# Step 6: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules.replace([float('inf'), float('-inf')], float('nan'), inplace=True)

# Convert frozenset columns to string
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

# Step 7: Save generated rules
rules.to_csv("GeneratedRules.csv", index=False)

# Step 8: Display results
print(f"Association rules generated with minSup = {minSup}")
print("\nGenerated Rules (Sample):")
print(rules.head())
