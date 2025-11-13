# Service Directory

This directory contains the FastAPI implementation for serving the trained model.

## Updated Progress

### Service Overview
This directory contains the FastAPI implementation for serving the trained model.

### Current Status
- Model saved as `baseline_model.pkl`.
- FastAPI service (`predict.py`) is ready for deployment.
- Dockerfile created for containerizing the FastAPI service.

### Next Steps
- Deploy the service to a cloud platform.

## Deployment Instructions

To manage dependencies and the environment, we use `uv`. Follow these steps to initialize and manage the environment:

### Procedure to Initialize the Project with `uv`

### Step 1: Navigate to the `service/` Folder
```bash
cd service
```

### Step 2: Initialize the Project with `uv`
Run the following command to initialize the project:
```bash
uv init
```
This will create a `pyproject.toml` file for managing dependencies.

### Step 3: Add Dependencies
- To add runtime dependencies (e.g., `scikit-learn`, `fastapi`, `uvicorn`), use:
  ```bash
  uv add scikit-learn fastapi uvicorn
  ```
- To add development dependencies (e.g., `requests`), use:
  ```bash
  uv add --dev requests
  ```

### Step 4: Lock Dependencies
After adding dependencies, `uv` will automatically generate a `uv.lock` file to lock the dependency versions.

### Step 5: Sync Dependencies
To ensure all dependencies are installed as per the lock file, run:
```bash
uv sync --locked
```

### Notes
- The `.venv` directory will be created automatically by `uv` in the `service/` folder.
- You do not need to manually activate the `.venv` environment; `uv` handles it automatically.
- If you need to activate the `.venv` manually for debugging, use:
  ```bash
  source .venv/bin/activate
  ```

### Running Locally
1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run the FastAPI Service**:
   ```bash
   uvicorn predict:app --host 0.0.0.0 --port 9696
   ```

### Running in a Docker Container
1. **Build the Docker Image**:
   ```bash
   docker build -t fastapi-service .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 9696:9696 fastapi-service
   ```

### Testing the `/predict` Endpoint

To test the `/predict` endpoint, use the following `curl` command:
```bash
curl -X POST http://0.0.0.0:9696/predict \
-H "Content-Type: application/json" \
-d '{
  "V17": -5.2,
  "V14": 2.3,
  "V12": -1.5,
  "V4": 0.8,
  "Amount": 123.45
}'
```

Expected response:
```json
{
  "prediction": 0,
  "probability": 7.186697320405055e-16
}
```

The service will be available at `http://localhost:9696`.
