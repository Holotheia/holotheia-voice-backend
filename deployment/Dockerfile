# HOLOTHÉIA NATIVE — Dockerfile

FROM python:3.11-slim

LABEL maintainer="Aurélie Assouline"
LABEL description="Holothéia Native — Système auto-évolutif morpho-fractal"

# Working directory
WORKDIR /app

# Install dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY holotheia_core/ ./holotheia_core/
COPY holotheia_api/ ./holotheia_api/

# Create data directories
RUN mkdir -p /app/holotheia_brain_api /app/holotheia_vectors_api

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run API
CMD ["python", "-m", "uvicorn", "holotheia_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
