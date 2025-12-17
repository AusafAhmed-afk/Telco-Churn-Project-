

# 📝 Telco Customer Churn Prediction API

A machine learning pipeline using **XGBoost** and **FastAPI** to predict customer churn risk based on service and contract data.

---

## 🚨 Problem Statement

Customer churn prediction is critical for businesses to retain valuable clients. This project predicts which customers are likely to discontinue their service. With these insights, companies can implement targeted retention strategies, reduce churn rates, and increase customer loyalty.

---

## 📊 Technical Stack

**Modeling & Data Processing**:

* **Python** – Programming language for model training and API development
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical computing
* **Scikit-learn** – Machine learning tools and utilities
* **XGBoost** – Gradient boosting model for churn prediction

**Deployment**:

* **FastAPI** – Asynchronous web framework for building APIs
* **Uvicorn** – ASGI server for running FastAPI

**Frontend/Demo Interface**:

* **Streamlit** – Interactive frontend for demoing the model

---

## 🏗️ Project Architecture

The project flow:

1. **Model** – Pre-trained model (`churn_model.pkl`) is loaded by the FastAPI server.
2. **Backend (FastAPI)** – Preprocesses input data, performs prediction using XGBoost, and returns the result.
3. **Frontend (Streamlit)** – Collects user input and displays prediction results in an interactive UI.

**Architecture Diagram:**

```
+----------------+     +--------------------+     +------------------------+
|  Streamlit UI  | --> |    FastAPI API     | --> |  XGBoost Model (Prediction)|
+----------------+     +--------------------+     +------------------------+
      (User)               (API)                      (Model)
```

---

## 🛠️ Setup & Installation

1. **Clone the repository**

```bash
git clone [YOUR_REPO_URL]
cd Telco_Churn_Prediction_Project
```

2. **Create and activate a virtual environment**

*Using conda:*

```bash
conda create --name churn-env python=3.9
conda activate churn-env
```

*Using venv:*

```bash
python -m venv churn-env
# On Linux/macOS
source churn-env/bin/activate
# On Windows
churn-env\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Train the model**

```bash
python src/model/train_model.py
```

This will generate:

* `churn_model.pkl` → trained model
* `feature_columns.json` → required feature columns

---

## 🚀 Running the Application

**Start Backend API**

```bash
uvicorn app.main:app --reload
```

* API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Start Frontend Demo**

```bash
streamlit run app_frontend.py
```

* Frontend UI: [http://localhost:8501](http://localhost:8501)

---

## 🧑‍💻 API Endpoint

### `/predict`

* **Method:** POST
* **Description:** Predicts the churn risk for a single customer.

**Request Body (example):**

```json
{
  "customer_id": 12345,
  "gender": "Female",
  "age": 45,
  "contract_type": "Month-to-month",
  "total_charges": 220.50,
  "service_type": "Fiber optic"
  ...
}
```

**Response Body (example):**

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.85
}
```

