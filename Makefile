#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ViT-for-101-food-app
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Descarga y extrae Food-101 (~5 GB)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m vit_for_101_food_app.dataset download

## Genera el split de validacion y el mapa de etiquetas
.PHONY: split
split:
	$(PYTHON_INTERPRETER) -m vit_for_101_food_app.dataset split

## Construye el cache de imagenes reescaladas
.PHONY: cache
cache:
	$(PYTHON_INTERPRETER) -m vit_for_101_food_app.dataset cache

## Verifica que los transforms coincidan con el AutoImageProcessor de cada modelo
.PHONY: verify
verify:
	$(PYTHON_INTERPRETER) -m vit_for_101_food_app.features verify

## Pipeline completo de preprocesamiento
.PHONY: preprocess
preprocess: data split cache verify


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
