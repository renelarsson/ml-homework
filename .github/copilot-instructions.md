# AI Coding Agent Instructions for `ml-homework`

## Project Overview
This repository contains assignments and projects related to machine learning, focusing on topics like regression, classification, and other fundamental ML concepts. The structure is organized by topic, with each folder representing a specific homework or project.

## Repository Structure
- **`01-intro/`**: Contains introductory materials and assignments.
- **`02-regression/`**: Focuses on regression techniques and related tasks.
- Additional folders may represent other topics as the repository grows.

## Key Developer Workflows
### Running Notebooks
1. Open the desired Jupyter Notebook (e.g., `02-regression/hw2.ipynb`).
2. Ensure the Python environment is properly configured.
3. Execute cells sequentially to complete the assignment.

### Testing Code
- If the repository includes test scripts, run them using:
  ```bash
  python -m unittest discover
  ```
- Ensure all dependencies are installed before running tests.

### Debugging
- Use print statements or debugging tools like `pdb` to step through code.
- For notebooks, leverage Jupyter's built-in debugging capabilities.

## Project-Specific Conventions
- Follow Python's PEP 8 style guide for code formatting.
- Use clear and descriptive variable names, especially for ML models and datasets.
- Document functions and classes with concise docstrings.

## Integration Points
- Ensure compatibility with common ML libraries like `numpy`, `pandas`, and `scikit-learn`.
- Verify that all dependencies are listed in a `requirements.txt` file.

## Examples
### Loading a Dataset
```python
import pandas as pd

data = pd.read_csv('data.csv')
print(data.head())
```

### Training a Model
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

## Notes
- Always validate your models using appropriate metrics.
- Keep datasets and outputs organized within their respective folders.

---

Feel free to update this file as the project evolves.