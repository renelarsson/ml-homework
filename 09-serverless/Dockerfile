FROM agrigorev/model-2025-hairstyle:v1

# Install required libraries
RUN pip install --no-cache-dir fastapi uvicorn torch torchvision onnxruntime

# Copy the FastAPI application code into the image
COPY lambda-function.py ./

# Copy the ONNX model file and its associated .data file into the image
COPY hair_classifier_empty.onnx ./
COPY hair_classifier_empty.onnx.data ./

# Override the ENTRYPOINT from the base image
ENTRYPOINT []

# Set the command to run the FastAPI application
CMD ["uvicorn", "lambda-function:app", "--host", "0.0.0.0", "--port", "8000"]