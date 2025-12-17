import pandas as pd
import pickle
import json
from pathlib import Path

# --- 1. Define Paths (CORRECTED) ---
# Since the script is at the project root, we only need .parent once.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "feature_columns.json"

# --- 2. Load model & feature columns ---
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully.")

    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)
        print("Feature columns loaded successfully.")

except FileNotFoundError as e:
    print("ERROR: Could not find model or feature files.")
    print(f"Expected to find files in: {MODEL_PATH.parent}")
    print("Please ensure you ran train_model.py successfully.")
    raise e

# --- 3. Sample Customer Data ---
customer_info = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 10,
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "MonthlyCharges": 75.0,
    "TotalCharges": 750.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "MultipleLines": "No",
    "PaymentMethod": "Electronic check"
}

df = pd.DataFrame([customer_info])

# --- 4. Preprocessing (Must match train_model.py) ---

# Map binary 'Yes/No' columns to 1/0
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
               'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']
binary_map = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0, 'Male': 0, 'Female': 1}

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

# One-Hot Encode remaining categorical columns
multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cat_cols)

# Ensure all columns from training are present (Alignment)
# Note: Using reindex is a cleaner way to align columns than concat
df_final = df.reindex(columns=feature_cols, fill_value=0)

# --- 5. Predict ---
pred = model.predict(df_final)[0]
probability = model.predict_proba(df_final)[0][1]

print("\n--- Prediction Result ---")
print("Churn Prediction:", "Yes" if pred == 1 else "No")
print(f"Churn Probability: {probability:.4f}")