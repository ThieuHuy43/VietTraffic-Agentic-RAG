.PHONY: build up down logs ingest clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ingest:
	docker compose exec backend_api python ingest.py

clean:
	docker compose down -v
	rm -rf data/qdrant_data/* checkpoints/*
