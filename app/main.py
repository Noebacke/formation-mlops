import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow
import mlflow.xgboost
import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_name = os.getenv("MLFLOW_MODEL_NAME")
    model_version = os.getenv("MLFLOW_MODEL_VERSION")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.xgboost.load_model(model_uri)
    yield


class PredictionRequest(BaseModel):
    chol: float
    crp: float
    phos: float


app = FastAPI(lifespan=lifespan, root_path="/proxy/8000")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with current model name and version.
    """
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "message": "NACE classifier",
        "model_name": f"{model_name}",
        "model_version": f"{model_version}",
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    data = np.array([[request.chol, request.crp, request.phos]])
    probabilities = model.predict_proba(data)[0]
    
    # Déterminer la classe prédite
    prediction = int(np.argmax(probabilities))  
    max_probability = float(probabilities[prediction])  # Probabilité associée

    return {
        "prediction": prediction,
        "probability": max_probability
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
