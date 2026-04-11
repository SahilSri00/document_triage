FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY openenv.yaml .
COPY src/ src/
COPY tasks/ tasks/
COPY server/ server/
COPY __init__.py .
COPY inference.py .
COPY README.md .

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check — matches OpenEnv's expected response
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, json; r=urllib.request.urlopen('http://localhost:7860/health'); d=json.loads(r.read()); assert d['status']=='healthy'" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
