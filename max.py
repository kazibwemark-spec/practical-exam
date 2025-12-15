# ===============================
# Part A: Data Preparation
# ===============================

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
data = {
    "Transaction_ID": [1,2,3,4,5,6,7,8,9,10],
    "Items": [
        "Bread, Milk, Eggs",
        "Bread, Butter",
        "Milk, Diapers, Beer",
        "Bread, Milk, Butter",
        "Milk, Diapers, Bread",
        "Beer, Diapers",
        "Bread, Milk, Eggs, Butter",
        "Eggs, Milk",
        "Bread, Diapers, Beer",
        "Milk, Butter"
    ]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Step 2: Convert items to transaction format
transactions = df["Items"].str.split(", ").tolist()
print("\nTransaction Format:")
for i, t in enumerate(transactions, start=1):
    print(f"Transaction {i}: {t}")

# Step 3: One-hot encoding for Apriori
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("\nOne-Hot Encoded Transactions:")
print(df_encoded)

# ===============================
# Part B: Apriori Algorithm
# ===============================

# Step 1: Generate frequent itemsets with minimum support = 0.2
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 2: Generate association rules with minimum confidence = 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print("\nAssociation Rules:")
print(rules)

# ===============================
# Part C: Interpretation
# ===============================

# Step 1: Identify top 3 rules based on lift
top_rules = rules.sort_values(by='lift', ascending=False).head(3)
print("\nTop 3 Strongest Rules Based on Lift:")
print(top_rules)

# Step 2: Business Insights
print("\nBusiness Insights / Recommendations:")
for i, row in top_rules.iterrows():
    print(f"- Rule: {set(row['antecedents'])} -> {set(row['consequents'])}")
    print(f"  Suggestion: Customers who buy {set(row['antecedents'])} also often buy {set(row['consequents'])}.")
    
