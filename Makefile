.PHONY: help install setup test lint format clean docker-up docker-down init-db train stream

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make setup        - Initial setup (create .env, download nltk data)"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code with black"
	@echo "  make init-db      - Initialize database"
	@echo "  make train        - Train model"
	@echo "  make stream       - Start Reddit stream"
	@echo "  make docker-up    - Start all services with Docker Compose"
	@echo "  make docker-down  - Stop all Docker services"
	@echo "  make clean        - Clean temporary files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

setup:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env file - please configure it"; fi
	@mkdir -p trained_models model_checkpoints data/raw data/processed

test:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

lint:
	flake8 src/ --max-line-length=127 --exclude=__pycache__
	mypy src/ --ignore-missing-imports || true

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

init-db:
	python scripts/init_db.py

train:
	python scripts/train_model.py --force

stream:
	@echo "Starting Reddit stream. Press Ctrl+C to stop."
	python scripts/stream_reddit.py --subreddits "technology" "artificial"

load-data:
	python scripts/load_sample_data.py --source both

docker-up:
	docker-compose up -d
	@echo "Services started. Access points:"
	@echo "  API: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Grafana: http://localhost:3000"
	@echo "  Prometheus: http://localhost:9090"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-rebuild:
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-celery:
	celery -A src.tasks.celery_app worker --loglevel=info

run-beat:
	celery -A src.tasks.celery_app beat --loglevel=info

all: install setup init-db load-data train
	@echo "Setup complete! Run 'make docker-up' to start services."
