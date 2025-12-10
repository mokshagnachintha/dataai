FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir openai pandas pydantic jinja2 sqlglot duckdb

ENV PYTHONUNBUFFERED=1

CMD ["python", "demo.py"]
