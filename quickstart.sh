#!/bin/bash

# Quick Start Script for Multi-Agent Fact-Checking System
# Phase 3: Production & Scaling

set -e

echo "=================================="
echo "Multi-Agent Fact-Checking System"
echo "Phase 3: Production & Scaling"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}IMPORTANT: Please edit .env and add your API keys!${NC}"
    echo ""
fi

# Function to start services
start_services() {
    echo -e "${GREEN}Starting services with Docker Compose...${NC}"
    docker-compose up -d

    echo ""
    echo "Waiting for services to be healthy..."
    sleep 5

    echo ""
    echo -e "${GREEN}Services started successfully!${NC}"
    echo ""
    echo "Access points:"
    echo "  - API:        http://localhost:8000"
    echo "  - API Docs:   http://localhost:8000/docs"
    echo "  - Metrics:    http://localhost:9090/metrics"
    echo "  - Prometheus: http://localhost:9091"
    echo "  - Grafana:    http://localhost:3000 (admin/admin)"
    echo ""
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped.${NC}"
}

# Function to view logs
view_logs() {
    echo -e "${GREEN}Viewing API logs (Ctrl+C to exit)...${NC}"
    docker-compose logs -f api
}

# Function to run tests
run_tests() {
    echo -e "${GREEN}Running tests...${NC}"
    docker-compose exec api pytest -v
}

# Function to check system health
check_health() {
    echo -e "${GREEN}Checking system health...${NC}"

    # Check API
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "  API:        ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  API:        ${RED}✗ Unhealthy${NC}"
    fi

    # Check Prometheus
    if curl -s http://localhost:9091 > /dev/null; then
        echo -e "  Prometheus: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Prometheus: ${RED}✗ Unhealthy${NC}"
    fi

    # Check Grafana
    if curl -s http://localhost:3000 > /dev/null; then
        echo -e "  Grafana:    ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Grafana:    ${RED}✗ Unhealthy${NC}"
    fi
}

# Main menu
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        ;;
    logs)
        view_logs
        ;;
    test)
        run_tests
        ;;
    health)
        check_health
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|test|health}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - View API logs"
        echo "  test    - Run test suite"
        echo "  health  - Check system health"
        exit 1
        ;;
esac
