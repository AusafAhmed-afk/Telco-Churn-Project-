from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import json
from pathlib import Path

app = FastAPI(title="Telco Churn Prediction API")

# --- 1. Define Paths (Correct for main.py inside 'app' folder) ---
# .parent.parent moves from /app/main.py to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "feature_columns.json"

# --- 2. Load Model & Feature Columns ---
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)
    print("Model and features loaded successfully.")
except FileNotFoundError:
    print("CRITICAL ERROR: Model or features file not found. Ensure train_model.py ran successfully.")
    model = None
    feature_cols = []


# --- 3. Pydantic Model (Input Schema) ---
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    PaperlessBilling: str
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    MultipleLines: str
    PaymentMethod: str


# --- 4. Prediction Endpoint (FINAL FIXED LOGIC) ---
@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    # Convert Pydantic object to DataFrame
    df = pd.DataFrame([customer.dict()])

    # ------------------ PREPROCESSING ----------------------

    # 1. Map binary columns to 1/0
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

    # Use 1.0/0.0 for safety, as XGBoost prefers float input.
    binary_map = {'Yes': 1.0, 'No': 0.0, 'No phone service': 0.0, 'No internet service': 0.0, 'Male': 0.0, 'Female': 1.0}

    for col in binary_cols:
        if col in df.columns:
            # Safely map to float
            df[col] = df[col].astype(str).map(binary_map).astype(float)

    # 2. One-Hot Encode remaining categorical columns (Correct Indentation)
    multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # 3. Align with training columns
    # Reindex aligns the columns, adds missing ones with fill_value=0.0
    df_aligned = df.reindex(columns=feature_cols, fill_value=0.0)

    # Final data type check: Ensure all values are floats.
    X_input = df_aligned.astype(float)

    # ------------------ PREDICTION ----------------------

    # CRITICAL FIX: Convert NumPy types to standard Python types for JSON encoding
    raw_pred = model.predict(X_input)[0]
    raw_probability = model.predict_proba(X_input)[0][1]

    pred_int = int(raw_pred)
    prob_float = float(raw_probability)

    return {
        "churn_prediction": "Yes" if pred_int == 1 else "No",
        "churn_probability": round(prob_float, 4)
    }