# Dockerfile

# Start from a Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install mlflow
RUN pip install mlflow

# Expose mlflow default port
EXPOSE 5000

# Run the mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
