<<<<<<< HEAD
📝 Telco Customer Churn Prediction API
Tagline

A machine learning pipeline that uses XGBoost and FastAPI to predict customer churn risk based on service and contract data.

Problem Statement

Customer churn prediction is critical for businesses to retain valuable clients. This project aims to predict which customers are likely to discontinue their service. The goal is to equip the company with insights that can help implement targeted retention strategies, reducing churn rates and increasing customer loyalty.

📊 Technical Stack

Modeling & Data Processing:

Python: Programming language for model training and API development.

Pandas: Data manipulation and analysis.

NumPy: Numerical computing for arrays and matrices.

Scikit-learn: Machine learning tools and utilities.

XGBoost: Gradient boosting model for churn prediction.

Deployment:

FastAPI: Asynchronous web framework for building APIs.

Uvicorn: ASGI server to run the FastAPI application.

Frontend/Demo Interface:

Streamlit: Framework for creating an interactive frontend interface.

🏗️ Project Architecture
Overview:

The flow of this project involves loading a pre-trained machine learning model, exposing an API via FastAPI, and using Streamlit for a user-friendly interface. Here's a breakdown of the architecture:

Model: The pre-trained model (churn_model.pkl) is loaded by the FastAPI server.

Frontend (Streamlit): The frontend collects user input and sends it to the FastAPI backend for prediction.

Backend (FastAPI): The server preprocesses the data, performs the prediction using XGBoost, and returns the churn risk result to the frontend.

Simple Diagram:
+----------------+     +--------------------+     +------------------------+
|  Streamlit UI  | --> |    FastAPI API     | --> |  XGBoost Model (Prediction)|
+----------------+     +--------------------+     +------------------------+
      (User)               (API)                      (Model)

🛠️ Setup and Installation

To run this project locally, follow these steps:

Clone the Repository:

git clone [YOUR_REPO_URL]
cd Telco_Churn_Prediction_Project


Create/Activate Environment:

For conda:

conda create --name churn-env python=3.9
conda activate churn-env


Or use venv for virtual environments:

python -m venv churn-env
source churn-env/bin/activate  # On Linux/macOS
churn-env\Scripts\activate     # On Windows


Install Dependencies:

pip install -r requirements.txt


Train the Model:
Run the following to generate the model file (churn_model.pkl) and the required feature columns (feature_columns.json):

python src/model/train_model.py

🚀 Running the Application
Start the Backend API:

In Terminal 1, run:

uvicorn app.main:app --reload


The API will be accessible at http://127.0.0.1:8000

API documentation will be available at http://127.0.0.1:8000/docs

Start the Frontend Demo:

In Terminal 2, run:

streamlit run app_frontend.py


The interactive UI will be available at http://localhost:8501

🧑‍💻 API Endpoints
/predict

Method: POST

Description: Predicts the churn risk for one customer.

Request Body:

The request should include customer data with 19 features, structured as a Pydantic model.

Example request body:

{
  "customer_id": 12345,
  "gender": "Female",
  "age": 45,
  "contract_type": "Month-to-month",
  "total_charges": 220.50,
  "service_type": "Fiber optic",
  ...
}

Response Body:

The response will indicate whether the customer is predicted to churn or not, along with the probability.

Example response body:

{
  "churn_prediction": "Yes",
  "churn_probability": 0.85
}
=======
# Data-Science-Projects-
Collection of my data science practice projects 
>>>>>>> 8595c4976478911be53324fdc80e8cad92577235
