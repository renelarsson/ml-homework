import joblib
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = joblib.load("best_model.pkl")
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("Model file not found. Ensure 'best_model.pkl' is in the correct directory.")
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
        probability = model.predict_proba(input_features)[0][1]

        # Apply threshold adjustment
        threshold = 0.92
        prediction = int(probability >= threshold)

        logger.info("Prediction made successfully.")
        return {
            "prediction": prediction,
            "probability": probability
        }
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return {"error": "An error occurred during prediction. Please check the input data."}