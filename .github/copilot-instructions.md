## AI coding agent guide for ml-homework

This repo is a set of ML Zoomcamp-style homeworks organized by topic (notebooks per module) plus FastAPI deployment examples for model serving.

### Layout and scope
- Notebooks by topic: `01-intro/hw1.ipynb`, `02-regression/hw2.ipynb`, `03-classification/hw3.ipynb`, `04-evaluation/hw4.ipynb`, `06-trees/hw6.ipynb`, `08-deep-learning/hw8.ipynb`, `09-serverless/hw9.ipynb`, `10-kubernetes/`.
- Deployment examples: `05-deployment/hw5/` (lead conversion prediction) and `07-midterm/project/service/` (fraud detection).
- Workshops: Guides in `workshops/` for deployment patterns.

### Core workflows
1) Notebooks
- Open target notebook and run cells top-to-bottom. Use Python 3.12+ with common ML libs (numpy, pandas, scikit-learn, matplotlib/seaborn).
- Data files colocated in module folders (e.g., `02-regression/car_fuel_efficiency.csv`) or downloaded from external sources (e.g., Kaggle for midterm).
- Kernel setup: Create venv with `uv venv`, activate, install deps with `uv pip install` or `uv sync` if uv.lock exists.

2) Local model serving (FastAPI)
- Service files: `predict.py` (dev entrypoint) loads pickled model and exposes POST `/predict`.
- Input schemas: Pydantic models in predict.py (e.g., Lead for hw5 with lead_source, number_of_courses_viewed, annual_income; InputData for midterm with V17,V14,V12,V10,V11).
- Response: Varies (e.g., {conversion_probability, convert} for hw5; {prediction, probability} for midterm).
- Run: `uvicorn predict:app --host 0.0.0.0 --port 9696`.
- Test: Use `test.py` for HTTP requests or `predict_single.py` for offline scoring.

3) Container build/run
- Dockerfiles use python:3.13-slim and uv for deps. Copy model artifact (.pkl/.bin) and app.py, run uvicorn on port 9696.
- Build: `docker build -t ml-service .` in service dir.
- Run: `docker run -p 9696:9696 ml-service`.

### Conventions and patterns
- Python version: `>=3.12` in pyproject.toml.
- Dependencies: Managed with uv; runtime deps include scikit-learn==1.6.1, fastapi, uvicorn; dev deps in [dependency-groups].
- Model artifacts: Pickled/joblib sklearn models saved as .pkl or .bin (e.g., `pipeline_v1.bin` in hw5, `best_model.pkl` in midterm).
- Two app entrypoints: `app.py` for container (minimal), `predict.py` for dev (includes logging, error handling).
- Training: Notebooks produce models; separate `train.py` in midterm for reproducibility.
- Testing: No unit tests; integration via HTTP probes or offline scripts.

### Examples
- Sample request for hw5: `{"lead_source": "organic_search", "number_of_courses_viewed": 4, "annual_income": 80304.0}`
- Response: `{"conversion_probability": 0.75, "convert": true}`
- Sample request for midterm: `{"V17": -0.5, "V14": 1.2, "V12": 0.8, "V10": -1.1, "V11": 0.3}`
- Response: `{"prediction": 0, "probability": 0.02}`

### Common pitfalls
- Missing artifact: Ensure model file is in service dir before uvicorn or docker build.
- Dependency resolver: Dockerfile uses uv with uv.lock; fallback to pip install from pyproject.toml if lock missing.
- Data acquisition: For midterm, download creditcard.csv from Kaggle and place in data/ (create if needed).
- Port: Services on 9696; update if changed.