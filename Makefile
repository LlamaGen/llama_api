# https://github.com/samuelcolvin/pydantic/blob/master/Makefile
.DEFAULT_GOAL := all
poetry = poetry run
isort = isort app/
black = black app/
mypy = mypy app/
flake8  = flake8 app/
pyupgrade = pyupgrade --py311-plus

# args := $(wordlist 2, 100, $(MAKECMDGOALS))

.PHONY: install-linting
install-linting:
	poetry add flake8 black isort mypy pyupgrade -G dev

.PHONY: install
install: install-linting
	pre-commit install
	@echo 'installed development requirements'

.PHONY: lint
lint: install-linting
	$(isort) --df --check-only
	$(black) --diff --check
	$(flake8)

.PHONY: format
format:
	$(poetry) $(pyupgrade)
	$(poetry) $(isort)
	$(poetry) $(black)
	$(poetry) $(mypy)
	$(poetry) $(flake8)

.PHONY: export-dependencies
export-dependencies:
	poetry export -f requirements.txt --output requirements.txt
	poetry export -f requirements.txt --output requirements-dev.txt --with=dev

.PHONY: database
database:
	docker run --name postgresql -e POSTGRES_USER=myusername -e POSTGRES_PASSWORD=mypassword -p 5432:5432 -d postgres

.PHONY: test-docker
test-docker:
	docker compose -f docker-compose.test.yaml up --build

.PHONY: test
test:
	poetry run pytest --cov=app --cov-report=html

.PHONY: migrate
migrate:
	@read -p "Enter migration message: " message; \
	poetry run alembic revision --autogenerate -m "$$message"

.PHONY: downgrade
downgrade:
	alembic downgrade -1

.PHONY: upgrade
upgrade:
	alembic upgrade +1

.PHONY: upgrade-offline
upgrade-offline:
	alembic upgrade head --sql

.PHONY: setup-model
setup-model:
	mkdir model
	wget https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/ggml-model-q4_K.gguf -P model

.PHONY: all
all: format export-dependencies
