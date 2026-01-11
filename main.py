from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# 🚀 Import trained ML model ONLY
from project import model

app = FastAPI(title="Loan Detector API")

# ✅ CORS CONFIG (Frontend ke liye VVIP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production me specific URL dena
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Pydantic Schema --------
class LoanData(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    bank_asset_value: float


# -------- Health Check --------
@app.get("/")
def home():
    return {"message": "Loan Detector API is running 🚀"}


# -------- Prediction API --------
@app.post("/predict")
def predict_loan(data: LoanData):

    # ✅ SAFE MANUAL ENCODING
    education = 1 if data.education.strip().lower() == "graduate" else 0
    self_employed = 1 if data.self_employed.strip().lower() == "yes" else 0

    # ✅ FEATURE ORDER MUST MATCH TRAINING
    features = np.array([[ 
        data.no_of_dependents,
        education,
        self_employed,
        data.income_annum,
        data.loan_amount,
        data.loan_term,
        data.cibil_score,
        data.bank_asset_value
    ]])

    # ✅ MODEL PREDICTION
    prediction = model.predict(features)[0]

    result = "Approved ✅" if prediction == 0 else "Rejected ❌"
    return {"prediction": result}
