FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*

COPY agents/ai_specialist/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ai_specialist /app

EXPOSE 8005

HEALTHCHECK --interval=20s --timeout=10s --start-period=20s --retries=5 \
  CMD curl -f http://localhost:8005/health || exit 1

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8005"]
