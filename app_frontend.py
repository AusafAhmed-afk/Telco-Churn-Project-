import streamlit as st
import requests
import json
import pandas as pd

# --- 1. FastAPI Endpoint URL ---
# The Streamlit app will send requests to this URL where your FastAPI server is running.
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(layout="wide", page_title="Telco Churn Prediction")

# --- Page Title and Header ---
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Use the controls on the left to input customer data and predict the likelihood of churn.")

## --- 2. Input Fields in Sidebar ---
st.sidebar.header("Customer Profile Input")
st.sidebar.markdown("---")


# Function to gather all 19 features required by your Pydantic model
def get_user_input():
    with st.sidebar.form("input_form"):
        # --- Basic/Demographic Inputs ---
        st.subheader("Demographics & Tenure")
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], help="0=No, 1=Yes")
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 1, 72, 12, help="How long the customer has stayed with the company.")

        st.markdown("---")
        st.subheader("Service & Payment")

        # --- Service & Billing Inputs ---
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method",
                                     ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                      "Credit card (automatic)"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        st.markdown("---")
        st.subheader("Internet Services (if applicable)")

        # --- Internet Services (using "No internet service" option) ---
        no_int_option = ["Yes", "No", "No internet service"]
        OnlineSecurity = st.selectbox("Online Security", no_int_option)
        OnlineBackup = st.selectbox("Online Backup", no_int_option)
        DeviceProtection = st.selectbox("Device Protection", no_int_option)
        TechSupport = st.selectbox("Tech Support", no_int_option)
        StreamingTV = st.selectbox("Streaming TV", no_int_option)
        StreamingMovies = st.selectbox("Streaming Movies", no_int_option)

        st.markdown("---")
        st.subheader("Charges")
        # --- Numerical Inputs ---
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0, step=0.1)
        # Note: TotalCharges must be consistent with MonthlyCharges * tenure, but we need the raw input
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=750.0, step=0.1,
                                       help="Total amount charged to the customer.")

        # The prediction button is part of the form
        submitted = st.form_submit_button("Predict Churn")

    # Compile data into the required JSON structure
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "PaperlessBilling": PaperlessBilling,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "Contract": Contract,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "MultipleLines": MultipleLines,
        "PaymentMethod": PaymentMethod
    }
    return data, submitted


input_data, submitted = get_user_input()

## --- 3. Prediction Logic ---

if submitted:
    st.header("Prediction Analysis")
    try:
        # 1. Send data to the FastAPI API
        response = requests.post(API_URL, json=input_data)

        # 2. Check for success (200 code)
        if response.status_code == 200:
            result = response.json()

            prediction = result['churn_prediction']
            probability = result['churn_probability']

            st.subheader(f"Final Prediction: {prediction}")

            # Display results with visual cues
            if prediction == "Yes":
                st.error(f"⚠️ HIGH CHURN RISK: The customer is likely to leave.")
            else:
                st.success(f"✅ STABLE CUSTOMER: The customer is likely to stay.")

            st.metric(label="Churn Probability", value=f"{probability:.1%}")

            with st.expander("Show Details"):
                st.dataframe(pd.Series(input_data).to_frame('Customer Data'))
                st.json(result)

        else:
            # Handle API errors (e.g., if FastAPI returns a 422 Validation Error)
            st.error(f"Error communicating with the API. Status Code: {response.status_code}")
            st.warning("Check the Uvicorn terminal for the detailed traceback.")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        # Handle case where the FastAPI server is not running
        st.error(f"Could not connect to the FastAPI server at {API_URL}.")
        st.warning("Please ensure your FastAPI application is running in a separate terminal.")