FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; httpx/tiktoken ship wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY dashboard ./dashboard
COPY scripts ./scripts

ENV DB_PATH=/data/observability.db
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
