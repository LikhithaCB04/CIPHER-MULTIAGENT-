FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY agents/ayeesha_fullstack/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ayeesha_fullstack /app

EXPOSE 8002

HEALTHCHECK --interval=20s --timeout=10s --start-period=20s --retries=5 \
  CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "ayeesha_agent:app", "--host", "0.0.0.0", "--port", "8002"]
