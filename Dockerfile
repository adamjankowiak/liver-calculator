FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models
COPY scripts ./scripts

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8000

CMD ["uvicorn", "liver_calculator.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
