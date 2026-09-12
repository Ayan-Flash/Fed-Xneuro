.PHONY: help install test test-unit test-integration run-api run-sim run-fedxneuro clean

help:
	@echo "Fed-XNeuro Makefile Commands:"
	@echo "  install          Install backend requirements"
	@echo "  test             Run entire test suite"
	@echo "  test-unit        Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  run-api          Start FastAPI backend server on port 8000"
	@echo "  run-sim          Run baseline FedAvg simulation"
	@echo "  run-fedxneuro    Run multimodal Fed-XNeuro simulation"
	@echo "  clean            Remove temporary and cache files"

install:
	pip install -r backend/requirements.txt

test:
	python -m pytest backend/tests/ -v

test-unit:
	python -m pytest backend/tests/unit/ -v

test-integration:
	python -m pytest backend/tests/integration/ -v

run-api:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

run-sim:
	python backend/run_simulation.py

run-fedxneuro:
	python backend/run_fedxneuro.py --clients 3 --rounds 3 --dp --explain

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
