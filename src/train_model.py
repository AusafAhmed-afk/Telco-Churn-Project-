import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from pathlib import Path
import json

# --- 1. Define Paths and Constants ---
# Use pathlib for robust path handling
BASE_DIR = Path(__file__).resolve().parent.parent # Gets the project root
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "churn_model.pkl"
# Change to .json for easier loading in FastAPI
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
DATA_PATH = BASE_DIR / "DATA" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# --- 2. Load and Clean Data ---
# Use the corrected path
data = pd.read_csv(DATA_PATH)

print(f"Loading data from: {DATA_PATH}")

# Data Cleaning
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])
data = data.drop('customerID', axis=1)

# Separate features by type for specific encoding
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
               'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies'] # Add other binary-like columns
multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
target_col = 'Churn'

# --- 3. Preprocessing (Encoding) ---

# Map binary 'Yes/No' columns to 1/0
binary_map = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
for col in binary_cols:
    # Handle 'gender' separately if needed, but the original dataset may only have 'Male/Female'
    if data[col].dtype == 'object':
        data[col] = data[col].map(binary_map).fillna(data[col].map({'Male': 0, 'Female': 1}))


# One-Hot Encode remaining categorical columns
data = pd.get_dummies(data, columns=multi_cat_cols, drop_first=True) # drop_first=True reduces multicollinearity

# Encode the target 'Churn' column (Yes=1, No=0)
data[target_col] = data[target_col].map({'Yes': 1, 'No': 0})

# --- 4. Features and Target ---
X = data.drop(target_col, axis=1)
y = data[target_col]

# Save feature columns as a JSON list
feature_cols = X.columns.tolist()
with open(FEATURES_PATH, "w") as f:
    json.dump(feature_cols, f)
print(f"Feature columns saved successfully to {FEATURES_PATH.name}")


# --- 5. Train/Test Split and Model Training ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_estimators=100, # Added for completeness
    learning_rate=0.1
)
model.fit(X_train, y_train)

# --- 6. Evaluate and Save ---
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved successfully to {MODEL_PATH.name}!")