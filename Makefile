.PHONY: venv install fmt lint test

venv:
	python -m venv .venv

install:
	python -m pip install -U pip
	pip install -r requirements.txt

fmt:
	python -m black .

lint:
	python -m ruff check .

test:
	pytest



