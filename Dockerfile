FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn

COPY 3_app.py .
COPY regression.joblib .

EXPOSE 8000

CMD ["uvicorn", "3_app:app", "--host", "0.0.0.0", "--port", "8000"]
