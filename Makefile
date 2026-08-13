.PHONY: install install-dev train-basic train-advanced inference-basic inference-advanced test lint docker-build docker-up api app

install:
	pip install -r requirements-prod.txt

install-dev:
	pip install -r requirements-dev.txt

train-basic:
	python entrypoint/train.py --mode basic --config config/basic/local.yaml

train-advanced:
	python entrypoint/train.py --mode advanced --config config/advanced/local.yaml

inference-basic:
	python entrypoint/inference.py --mode basic --config config/basic/local.yaml

inference-advanced:
	python entrypoint/inference.py --mode advanced --config config/advanced/local.yaml

test:
	pytest src/tests/ -v

lint:
	black --check src/ entrypoint/ api/
	flake8 src/ entrypoint/ api/

docker-build:
	docker build -t heart-disease-prediction .

docker-up:
	docker-compose up --build

api:
	uvicorn api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py
