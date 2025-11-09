.PHONY: build up down restart clean logs

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart: down build up

clean:
	docker compose down --rmi local
	docker rmi mongo:6.0 apache/kafka:4.1.0 apache/spark
	docker image prune -f

logs:
	docker compose logs