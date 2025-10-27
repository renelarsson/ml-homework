FROM python:3.13.5-slim-bookworm

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies using uv
RUN uv sync --locked

# Explicitly install uvicorn, fastapi, and scikit-learn globally
RUN pip install uvicorn fastapi scikit-learn

# Copy application files
COPY app.py pipeline_v1.bin ./

# Expose the port
EXPOSE 9696

# Start the FastAPI application
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]