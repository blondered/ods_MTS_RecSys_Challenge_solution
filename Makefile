all:
	@echo "Running isort..."
	@isort src
	@echo "Running black..."
	@black src/*/*.py

lint:
	@echo "Running pylint..."
	@pylint src
	@echo "Running mypy..."
	@mypy src
	@echo "Running flake8..."
	@flake8 src

