# FastAPI script:
# 1. Loads the pipeline_v1.bin model.
# 2. Defines a /predict endpoint to accept client data and return the conversion probability.

import pickle
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn

# Define the input schema
class Lead(BaseModel):
    lead_source: Literal["paid_ads", "organic_search"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    conversion_probability: float
    convert: bool

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Initialize FastAPI app
app = FastAPI(title="lead-conversion-prediction")

def predict_single(lead):
    result = pipeline.predict_proba([lead])[0, 1]
    return float(result)

@app.post("/predict")
def predict(lead: Lead) -> PredictResponse:
    prob = predict_single(lead.dict())

    return PredictResponse(
        conversion_probability=prob,
        convert=prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)