FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY MealHistory_UTF8.csv ./
RUN pip install --no-cache-dir .

ENV MEAL_HISTORY_DATA=/app/MealHistory_UTF8.csv
EXPOSE 8000
CMD ["meal-history-api"]
