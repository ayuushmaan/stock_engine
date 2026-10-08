# Stock Engine — Production & Research Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential, git, libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python package dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-cov

# Copy project source code
COPY config/ ./config/
COPY pipeline/ ./pipeline/
COPY models/ ./models/
COPY research/ ./research/
COPY tests/ ./tests/
COPY README.md .env.example* ./

# Install project package in editable mode
RUN pip install --no-cache-dir -e .

# Expose Streamlit default port if dashboard is run
EXPOSE 8501

# Default command: run pytest suite
CMD ["pytest", "tests/", "-v"]
