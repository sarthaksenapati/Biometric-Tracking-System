.PHONY: help build up down restart logs clean dev-linux dev-windows

help:
	@echo "Biometric Tracking System - Docker Commands"
	@echo "=========================================="
	@echo "build          - Build all Docker images"
	@echo "up              - Start all services (cross-platform)"
	@echo "up-linux        - Start with Linux camera support"
	@echo "up-windows      - Start with Windows config (tracker on host)"
	@echo "down           - Stop all services"
	@echo "restart        - Restart all services"
	@echo "logs           - View logs from all services"
	@echo "logs-tracker   - View tracker logs"
	@echo "logs-backend    - View backend logs"
	@echo "logs-dashboard - View dashboard logs"
	@echo "clean          - Stop and remove all containers/volumes"
	@echo "dev-linux      - Development mode with hot-reload (Linux)"
	@echo "dev-windows    - Development mode (Windows)"
	@echo "ps             - Show running containers"

build:
	docker-compose build

up:
	docker-compose up -d

up-linux:
	docker-compose -f docker-compose.yml -f docker-compose.linux.yml up -d

up-windows:
	docker-compose -f docker-compose.yml -f docker-compose.windows.yml up -d

down:
	docker-compose down

restart: down up

logs:
	docker-compose logs -f

logs-tracker:
	docker-compose logs -f tracker

logs-backend:
	docker-compose logs -f backend

logs-dashboard:
	docker-compose logs -f dashboard

clean:
	docker-compose down -v --rmi all

dev-linux:
	docker-compose -f docker-compose.yml -f docker-compose.linux.yml -f docker-compose.dev.yml up -d

dev-windows:
	docker-compose -f docker-compose.yml -f docker-compose.windows.yml -f docker-compose.dev.yml up -d

ps:
	docker-compose ps

# Quick command to start tracker on host (Windows/Mac)
run-tracker-host:
	python run_tracker_multi.py
