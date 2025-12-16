import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse

# -------------------------------
# Load model from MLflow Registry
# -------------------------------
MODEL_URI = "models:/CreditRiskModel/Production"

model = mlflow.sklearn.load_model(MODEL_URI)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Credit Risk Scoring API")

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    risk_prob = model.predict_proba(X)[0][1]

    return PredictionResponse(risk_probability=risk_prob)
