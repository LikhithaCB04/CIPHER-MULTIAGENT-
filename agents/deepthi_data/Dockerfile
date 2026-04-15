# Dockerfile — Deepthi's Data Science Agent
# Likitha uses this in docker-compose.yml

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deepthi_agent.py .

EXPOSE 8001

# Health check — used by docker-compose
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "deepthi_agent:app", "--host", "0.0.0.0", "--port", "8001"]
