import joblib
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = joblib.load("baseline_model.pkl")
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("Model file not found. Ensure 'baseline_model.pkl' is in the correct directory.")
    raise
except Exception as e:
    logger.error(f"An error occurred while loading the model: {e}")
    raise

# Define the FastAPI app
app = FastAPI()

# Define the input schema
class InputData(BaseModel):
    V17: float
    V14: float
    V12: float
    V4: float
    Amount: float

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to a format suitable for the model
        input_features = [[data.V17, data.V14, data.V12, data.V4, data.Amount]]
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)[0][1]
        logger.info("Prediction made successfully.")
        return {
            "prediction": int(prediction[0]),
            "probability": probability
        }
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return {"error": "An error occurred during prediction. Please check the input data."}