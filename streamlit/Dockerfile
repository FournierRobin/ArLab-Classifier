FROM python:3.9-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY app.py .

# Set the entrypoint command
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]