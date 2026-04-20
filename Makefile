.PHONY: install lint test api train

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
