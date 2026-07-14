FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY orchestrator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator /app/orchestrator
WORKDIR /app/orchestrator

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --start-period=20s --retries=5 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "orchestrator.py"]
