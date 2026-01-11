import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df = df.drop([
    "loan_id",
    "commercial_assets_value",
    "luxury_assets_value",
    "residential_assets_value"
], axis=1)

df = df.dropna()

# ---------------------------
# 2. Encode Categorical Data
# ---------------------------
le_education = LabelEncoder()
le_self = LabelEncoder()
le_status = LabelEncoder()

df["education"] = le_education.fit_transform(df["education"])
df["self_employed"] = le_self.fit_transform(df["self_employed"])
df["loan_status"] = le_status.fit_transform(df["loan_status"])

# ---------------------------
# 3. Split Data
# ---------------------------
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 4. Train Model
# ---------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------
# 5. Model Accuracy
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# =================================================
# 6. MANUAL USER INPUT (CLI UI)
# =================================================
print("\n--- Enter Applicant Details ---")
