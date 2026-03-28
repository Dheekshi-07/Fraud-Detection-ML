import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------------
# Step 1: Load dataset
# ----------------------------
df = pd.read_csv("fraud_data.csv")  # Make sure this CSV is in the same folder

print("Dataset preview:")
print(df.head())

# ----------------------------
# Step 2: Average transaction amount
# ----------------------------
print("\nAverage Amount:", np.mean(df["Amount"]))

# ----------------------------
# Step 3: Encode categorical columns
# ----------------------------
le_type = LabelEncoder()
df["Transaction_Type"] = le_type.fit_transform(df["Transaction_Type"])

# Explicitly map Fraud to 0/1
df["Fraud"] = df["Fraud"].map({"No": 0, "Yes": 1})

# Check counts to ensure mapping is correct
print("\nFraud value counts:")
print(df["Fraud"].value_counts())

# ----------------------------
# Step 4: Visualize Fraud vs Non-Fraud
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Fraud", data=df, palette=["skyblue", "orange"])
plt.xticks([0,1], ["Non-Fraud (0)", "Fraud (1)"])
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()

# ----------------------------
# Step 5: Prepare data for ML
# ----------------------------
X = df.drop("Fraud", axis=1)
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 6: Train Logistic Regression
# ----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------------
# Step 7: Make predictions and evaluate
# ----------------------------
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("\nModel Accuracy:", accuracy)