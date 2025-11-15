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

## Exploratory Data Analysis (EDA)

### Summary of Findings
- **Distributions**: Most features are scaled, with no significant outliers.
- **Correlations**: Strong correlations observed between `V14` and the target variable.
- **Outliers**: Minimal outliers detected in `Amount`.

### EDA Findings
- **Distributions**: Key features such as `V17`, `V14`, and `Amount` were analyzed for their distributions.
- **Correlations**: Strong correlations were observed between `V14` and the target variable.
- **Outliers**: Outliers were identified and handled appropriately.

### Preprocessing Steps
1. Handled missing values by imputing the median.
2. Scaled numerical features using MinMaxScaler.
3. Encoded categorical variables using one-hot encoding.
- Standard scaling was applied to numerical features.
- Categorical features were one-hot encoded.

## Modeling and Tuning

### Alternative Models Tested
1. Logistic Regression (Baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier

### Final Model Selection
- **Model**: Weighted Logistic Regression
- **Rationale**: Best balance of precision and recall.

### Cross-Validation Scores
| Model                     | Precision | Recall | F1-Score | ROC AUC |
|---------------------------|-----------|--------|----------|---------|
| Logistic Regression       | 0.72      | 0.65   | 0.68     | 0.91    |
| Random Forest Classifier  | 0.78      | 0.70   | 0.74     | 0.94    |
| Gradient Boosting Classifier | 0.80   | 0.72   | 0.76     | 0.96    |
| Logistic Regression| 0.85     | 0.82      | 0.80   |
| Random Forest      | 0.88     | 0.85      | 0.83   |
| Gradient Boosting  | 0.90     | 0.87      | 0.86   |

## Final Model
- **Model**: Weighted Logistic Regression
- **Threshold**: 0.92
- **Performance**:
  - **Validation Set**:
    - Precision: 0.84
    - Recall: 0.76
    - F1-Score: 0.80
    - ROC AUC: 0.98
  - **Test Set**:
    - Precision: 0.72
    - Recall: 0.82
    - F1-Score: 0.76
    - ROC AUC: 0.982

## Training the Model

To train the model from scratch, run the following command:
```bash
python train.py
```
To train the model from scratch, run the following command:
```bash
python src/train.py
```

## Workflow Diagram

```plaintext
Dataset → Preprocessing → Model Training → API Deployment
```

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

## Deploying to Fly.io

### Prerequisites
1. Install the Fly.io CLI:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```
2. Log in to Fly.io:
   ```bash
   flyctl auth login
   ```

### Deployment Steps
1. Initialize the Fly.io app:
   ```bash
   flyctl launch
   ```
   - Choose a unique app name.
   - Select a region close to your users.
   - Do not deploy immediately.

2. Update the `fly.toml` file:
   - Ensure `internal_port` is set to `9696` under `[http_service]`.

3. Build and deploy the app:
   ```bash
   flyctl deploy
   ```

4. Verify the app is running:
   ```bash
   flyctl status
   ```

5. Check logs for errors:
   ```bash
   flyctl logs
   ```

### Testing the Deployment
1. Test the root endpoint:
   ```bash
   curl https://<your-app-name>.fly.dev/
   ```
2. Test the `/predict` endpoint:
   ```bash
   curl -X POST https://<your-app-name>.fly.dev/predict \
   -H "Content-Type: application/json" \
   -d '{ "V17": -5.2, "V14": 2.3, "V12": -1.5, "V4": 0.8, "Amount": 123.45}'
   ```