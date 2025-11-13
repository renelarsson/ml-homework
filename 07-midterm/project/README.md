# Midterm Project: Fraud Detection in Credit Card Transactions

## Project Description
This project focuses on detecting fraudulent credit card transactions using publicly available datasets. The goal is to build a machine learning model that can accurately classify transactions as fraudulent or legitimate, helping financial institutions minimize losses due to fraud.

## Dataset
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Structure**:
  - Features: Numerical features representing transaction details.
  - Target: A binary variable indicating fraud (1) or legitimate (0).
- **Size**: ~284,807 transactions with 492 fraud cases (~0.17% fraud rate).
- **Format**: CSV file.

## Dataset Acquisition

To download the dataset:

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Download the `creditcard.csv` file.
3. Place the file in the `data/` directory.

### Notes
- Ensure the dataset is unzipped and accessible in the `data/` directory.
- Document any preprocessing steps in the `notebooks/` directory.

## Repository Structure
- `data/` 
  - Raw and processed data.
- `notebooks/` 
  - Jupyter notebooks for EDA and modeling.
- `src/` 
  - Source code for data processing and model training.
- `service/` 
  - FastAPI implementation for serving the model.

# Project Setup

## Environment Management

To manage dependencies and the environment, we use `uv`. Follow these steps to initialize and manage the environment:

### Initialize the Project
1. Navigate to the `service/` folder:
   ```bash
   cd service
   ```
2. Run the following command to initialize the project:
   ```bash
   uv init
   ```

### Adding Dependencies
- To add runtime dependencies, use:
  ```bash
  uv add scikit-learn fastapi uvicorn
  ```
- To add development dependencies, use:
  ```bash
  uv add --dev requests
  ```

### Notes
- The `uv.lock` file will be generated to lock dependencies.
- Use `uv` commands to ensure consistent dependency management across environments.

## Running the Service

### Locally
1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run the FastAPI Service**:
   ```bash
   uvicorn predict:app --host 0.0.0.0 --port 9696
   ```

### Docker
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