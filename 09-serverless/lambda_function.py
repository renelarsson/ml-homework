from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime
import numpy as np
from PIL import Image

app = FastAPI()

# Load the ONNX model
session = onnxruntime.InferenceSession("hair_classifier_empty.onnx")

class ImageInput(BaseModel):
    image: list  # Replace with the appropriate input schema

@app.post("/predict")
def predict(input: ImageInput):
    # Preprocess the input
    img_array = np.array(input.image)
    # Add batch dimension and preprocess
    img_array = img_array[np.newaxis, :, :, :]
    # Convert input data to float32
    img_array = img_array.astype(np.float32)
    # Log the data type for debugging
    print(f"Input data type after conversion: {img_array.dtype}")
    
    # Ensure the input tensor has the correct rank (4D)
    if img_array.ndim == 5:
        img_array = img_array.squeeze(axis=0)  # Remove the extra dimension
    elif img_array.ndim != 4:
        raise ValueError(f"Invalid input shape: {img_array.shape}. Expected a 4D tensor.")
        
    # Run the model
    outputs = session.run(None, {"input": img_array})
    return {"prediction": outputs[0].tolist()}