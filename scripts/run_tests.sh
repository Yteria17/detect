#!/bin/bash
# Test runner script for Multi-Agent Fact-Checking System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Multi-Agent Fact-Checking System - Test Runner ===${NC}\n"

# Parse arguments
RUN_COVERAGE=true
RUN_BENCHMARKS=false
TEST_PATH="tests/"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cov)
            RUN_COVERAGE=false
            shift
            ;;
        --benchmarks)
            RUN_BENCHMARKS=true
            shift
            ;;
        --path)
            TEST_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./scripts/run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-cov        Run tests without coverage"
            echo "  --benchmarks    Run benchmark tests"
            echo "  --path PATH     Run specific test file or directory"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install -r requirements.txt"
    exit 1
fi

# Run tests
if [ "$RUN_COVERAGE" = true ]; then
    echo -e "${YELLOW}Running tests with coverage...${NC}\n"
    pytest "$TEST_PATH" \
        --cov=agents \
        --cov=api \
        --cov=monitoring \
        --cov=utils \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-fail-under=70 \
        -v

    echo -e "\n${GREEN}Coverage report generated in htmlcov/index.html${NC}"
else
    echo -e "${YELLOW}Running tests without coverage...${NC}\n"
    pytest "$TEST_PATH" -v
fi

# Run benchmarks if requested
if [ "$RUN_BENCHMARKS" = true ]; then
    echo -e "\n${YELLOW}Running benchmarks...${NC}\n"
    python tests/benchmarks.py
fi

echo -e "\n${GREEN}✓ All tests completed successfully!${NC}"
