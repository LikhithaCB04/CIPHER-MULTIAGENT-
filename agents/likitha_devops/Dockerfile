# Use a slim version of Python to keep the container small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your agent's code into the container
COPY agent.py .

# Run the agent on your assigned port (8004)
# We use 0.0.0.0 so you can access it from your laptop
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8004"]
