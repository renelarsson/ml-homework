# Midterm Project (Scaffold)

This folder will contain the full midterm project: dataset, notebooks, training script, API service, and Docker image.

## Status
- Planning: in-progress (see `plan.md`)
- Code: not started

## Project outline
- Problem: TBD (to be selected based on current industry demand)
- Dataset: TBD
- Target: TBD
- Metric(s): TBD (classification: ROC AUC / F1; regression: RMSE / MAE — as appropriate)

## Repository structure (planned)
- `data/` — raw and processed data (or scripts/links to download)
- `notebooks/` — exploration, EDA, modeling
- `src/` — reusable Python code (feature prep, training utils)
- `service/` — FastAPI service (`serve.py`), schema models, example requests
- `train.py` — trains the final model and saves an artifact (e.g., `model.bin`)
- `predict.py` — loads the artifact and scores a single record
- `Dockerfile` — container to run the service locally
- `pyproject.toml` or `requirements.txt` — dependencies

## How this will run (once implemented)
1) Create and activate a Python 3.12+ environment
2) Install dependencies (pyproject or requirements)
3) Reproduce training:
   - Run `train.py` to produce the model artifact
4) Start the API service locally (FastAPI):
   - Run `uvicorn service.serve:app --host 0.0.0.0 --port 9696`
5) Send a request to `/predict` (JSON payload) and receive a prediction
6) Build & run Docker image for local testing

Concrete commands and example requests will be added after the dataset and problem are finalized.

## Evaluation readiness
This project will align with the midterm checklist:
- Clear problem and dataset description
- Reproducible EDA and modeling
- Scripted training + working web service
- Dockerized service with usage instructions
- Brief limitations and next steps

See `../project-tips.md` and `../README.md` for the course rubric and best practices.
