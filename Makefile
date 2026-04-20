.PHONY: install lint test api train docker-build docker-up docker-down

install:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev,train]

lint:
	ruff check src tests scripts

test:
	pytest

api:
	python scripts/serve_api.py

train:
	python scripts/train_meta_logistic.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down
