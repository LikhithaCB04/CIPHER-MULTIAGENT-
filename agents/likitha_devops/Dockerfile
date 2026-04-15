# Use a slim version of Python to keep the container small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your agent's code into the container
COPY agent.py .

# Run the agent on your assigned port (8004)
# We use 0.0.0.0 so you can access it from your laptop
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8004"]
