.PHONY: help install lint format test test-unit test-integration run docker-build docker-up docker-down clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/

test: ## Run all tests
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	python -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	python -m pytest tests/integration/ -v --tb=short

run: ## Run the application locally
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t ml-drift-monitor:latest .

docker-up: ## Start all services with Docker Compose
	docker compose -f docker/docker-compose.yml up -d --build

docker-down: ## Stop all Docker Compose services
	docker compose -f docker/docker-compose.yml down

docker-logs: ## View Docker Compose logs
	docker compose -f docker/docker-compose.yml logs -f

clean: ## Clean build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ dist/ build/ *.egg-info
