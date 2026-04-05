FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY websocket_manager.py .
COPY dashboard.py .
COPY main.py .
COPY dashboard.html .
COPY data-viewer.html .
COPY capture.html .
COPY calibration.html .
COPY processing_api.py .
COPY calibration_api.py .
COPY processing/ ./processing/
COPY static/ ./static/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
