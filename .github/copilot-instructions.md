## AI coding agent guide for ml-homework

This repo is a set of ML Zoomcamp-style homeworks organized by topic (notebooks per module) plus a minimal FastAPI deployment example. Agents should focus on two workflows: running notebooks for exploration and running the deployment service for serving a trained model.

### Layout and scope
- Notebooks by topic: `01-intro/hw1.ipynb`, `02-regression/hw2.ipynb`, `03-classification/hw3.ipynb`, `04-evaluation/hw4.ipynb`, `06-trees/hw6.ipynb`.
- Midterm guidance: `07-midterm/README.md` (project expectations and deliverables).
- Deployment example: `05-deployment/hw5/` (FastAPI service + Dockerfile) that serves a pickled sklearn pipeline saved as `pipeline_v1.bin`.

### Core workflows
1) Notebooks
- Open the target notebook and run top-to-bottom. Use Python 3.12+ and common ML libs (numpy, pandas, scikit-learn). Data files are colocated in the module folders (e.g., `02-regression/car_fuel_efficiency.csv`).
- If you need a kernel: create/activate a venv and install minimal deps: numpy, pandas, scikit-learn, matplotlib/seaborn as needed.

2) Local model serving (FastAPI)
- Service files: `05-deployment/hw5/app.py` (container entrypoint) and `05-deployment/hw5/predict.py` (dev entrypoint). Both load `pipeline_v1.bin` from the working directory and expose POST `/predict`.
- Input schema (Pydantic):
  - `lead_source: "paid_ads" | "organic_search"`
  - `number_of_courses_viewed: int >= 0`
  - `annual_income: float >= 0.0`
- Response shape: `{ "conversion_probability": float, "convert": bool }` (threshold 0.5).
- Quick dev run: start with uvicorn against `predict.py` and send a POST to `/predict`. `05-deployment/hw5/test.py` contains a working request example.

3) Container build/run
- Dockerfile: `05-deployment/hw5/Dockerfile` uses python:3.13-slim and uv (Astral) for dependency sync. It copies `app.py` and `pipeline_v1.bin` and runs `uvicorn app:app --port 9696`.
- Note: The Dockerfile references `uv.lock` and `.python-version`. If these files are missing, either generate them with uv, or simplify the image to install from `pyproject.toml` (pip install) before copying `app.py`.

### Conventions and patterns
- Python version: `pyproject.toml` in `05-deployment/hw5` sets `requires-python = ">=3.12"`; deps include `scikit-learn==1.6.1`, `fastapi`, `requests`.
- Training artifacts: the API expects `pipeline_v1.bin` (pickle of an sklearn Pipeline) to be present at service start; notebooks or a separate script should produce it.
- Two app entrypoints: container uses `app.py`; local dev commonly uses `predict.py` (declares `app` and has a `__main__` uvicorn runner). Keep both reading from the same artifact for parity.
- Testing: there are no unit tests; use `05-deployment/hw5/test.py` for an end-to-end HTTP probe, or `predict_single.py` for offline scoring.

### Examples
- Sample request body:
  `{ "lead_source": "organic_search", "number_of_courses_viewed": 4, "annual_income": 80304.0 }`
- Expected response keys: `conversion_probability` (float), `convert` (bool).

### Common pitfalls (read first)
- Missing artifact: if `pipeline_v1.bin` isn’t in `05-deployment/hw5/` at runtime, the app will fail on import. Generate or copy the file before running uvicorn or building the image.
- Dependency resolver mismatch: the Dockerfile assumes uv with `uv.lock`; when absent, prefer a fallback flow (pip install from `pyproject.toml`) or add a lock file.
- Port: service listens on `9696`; update clients/tests if you change it.

Keep this file concise and specific to this repo. Update it when service contracts, entrypoints, or dependency tooling change.