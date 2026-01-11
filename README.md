# Loan Approval Prediction Backend (FastAPI)

This backend is built using **FastAPI** to serve a trained machine learning model for loan approval prediction.  
The model is imported from a separate module and used to provide fast and reliable predictions via REST APIs.  
FastAPI ensures high performance, automatic validation, and interactive API documentation.  
The backend is designed to integrate seamlessly with a frontend application.  
It is deployed on **Render** for public access.

## 🚀 Tech Stack
- Python
- FastAPI
- Scikit-learn
- NumPy, Pandas
- Uvicorn / Gunicorn

## 🔗 API Endpoints

### Health Check
**GET /**  
Returns a message confirming the backend is running.

### Prediction
**POST /predict**  
Accepts loan-related data in JSON format and returns the prediction result.

Example request:
```json
{
  "no_of_dependents": 2,
  "income_annum": 500000,
  "loan_amount": 200000,
  "loan_term": 12,
  "cibil_score": 750,
  "bank_asset_value": 300000
}
