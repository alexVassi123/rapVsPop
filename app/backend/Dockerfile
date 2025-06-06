FROM python:3.10-slim AS builder

WORKDIR /app

# Copy requirements from backend folder
COPY app/backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy backend source code
COPY app/backend/ ./app/backend/

# Copy inference folder (model)
COPY inference/ ./inference/

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
