FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ /app/src/
RUN pip install --no-cache-dir .

# Tạo thư mục chứa dữ liệu runtime thay vì hardcode COPY file CSV
RUN mkdir -p /app/data

ENV DATA_DIR=/app/data
EXPOSE 8000

CMD ["uvicorn", "meal_history.web:app", "--host", "0.0.0.0", "--port", "8000"]