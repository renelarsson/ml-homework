from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Define the FastAPI app
app = FastAPI()

# Load the pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

# Define the input schema
class ClientData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Define the /predict endpoint
@app.post("/predict")
def predict(client: ClientData):
    client_dict = client.dict()
    prediction = pipeline.predict_proba([client_dict])[0, 1]
    # Convert numpy types to native Python types so FastAPI can JSON-encode them
    prob = float(prediction)
    will_convert = bool(prob >= 0.5)

    result = {
        "conversion_probability": prob,
        "convert": will_convert
    }
    return result