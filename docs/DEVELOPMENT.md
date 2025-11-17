# Development Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Code Structure](#code-structure)
3. [Coding Standards](#coding-standards)
4. [Development Workflow](#development-workflow)
5. [Adding Features](#adding-features)
6. [Debugging](#debugging)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)

---

## Getting Started

### Development Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/detect.git
cd detect

# Create development branch
git checkout -b feature/your-feature-name

# Setup virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies with development tools
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Copy environment file
cp .env.example .env.development
```

### Development Dependencies

```txt
# requirements-dev.txt

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
pytest-mock==3.12.0
hypothesis==6.92.2

# Code Quality
black==23.12.1
flake8==7.0.0
mypy==1.8.0
pylint==3.0.3
isort==5.13.2

# Pre-commit
pre-commit==3.6.0

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0
myst-parser==2.0.0

# Profiling
py-spy==0.3.14
memory-profiler==0.61.0

# Debugging
ipdb==0.13.13
pdbpp==0.10.3
```

---

## Code Structure

### Project Layout

```
detect/
├── detect/                    # Main package
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry
│   ├── agents/               # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py           # BaseAgent class
│   │   ├── collector.py      # Agent 1
│   │   ├── classifier.py     # Agent 2
│   │   ├── anomaly.py        # Agent 3
│   │   ├── fact_checker.py   # Agent 4
│   │   └── reporter.py       # Agent 5
│   ├── api/                  # API endpoints
│   │   ├── v1/
│   │   │   ├── fact_check.py
│   │   │   ├── batch.py
│   │   │   └── monitoring.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── rate_limit.py
│   ├── services/             # Core services
│   │   ├── llm.py            # LLM service
│   │   ├── retrieval.py      # RAG retrieval
│   │   ├── embeddings.py     # Embedding service
│   │   └── credibility.py    # Source scoring
│   ├── models/               # Data models
│   │   ├── claim.py
│   │   ├── evidence.py
│   │   └── verdict.py
│   ├── db/                   # Database
│   │   ├── postgres.py
│   │   ├── redis.py
│   │   └── weaviate.py
│   ├── orchestration/        # Workflow orchestration
│   │   ├── langgraph.py
│   │   └── patterns.py
│   ├── utils/                # Utilities
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── text.py
│   └── config/               # Configuration
│       └── settings.py
├── tests/                    # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
├── config/                   # Configuration files
├── docker/                   # Docker files
└── examples/                 # Usage examples
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `agents/` | Individual agent logic |
| `api/` | HTTP endpoints and middleware |
| `services/` | Reusable business logic |
| `models/` | Data structures and validation |
| `db/` | Database connections and queries |
| `orchestration/` | Workflow management |
| `utils/` | Helper functions |

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# Line length: 100 characters (not 79)
# Use double quotes for strings
# Use trailing commas in multi-line structures

# Good
def process_claim(
    claim: str,
    options: Dict[str, Any],
    context: Optional[Dict] = None,
) -> FactCheckResult:
    """
    Process a claim and return verification result.

    Args:
        claim: The claim text to verify
        options: Processing options
        context: Optional context information

    Returns:
        FactCheckResult with verdict and evidence

    Raises:
        ValueError: If claim is invalid
        ProcessingError: If verification fails
    """
    if not claim or len(claim) < 10:
        raise ValueError("Claim must be at least 10 characters")

    return FactCheckResult(verdict="SUPPORTED", confidence=0.87)
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Union, Tuple

def retrieve_evidence(
    query: str,
    max_results: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """Retrieve evidence documents"""
    pass

# For complex types, use TypedDict
from typing_extensions import TypedDict

class ClaimInput(TypedDict):
    text: str
    source_url: Optional[str]
    priority: str
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_confidence(
    verdicts: List[str],
    evidence_scores: List[float]
) -> float:
    """
    Calculate overall confidence score from individual verdicts.

    Args:
        verdicts: List of verdict strings (SUPPORTED/REFUTED/INSUFFICIENT_INFO)
        evidence_scores: Confidence scores for each piece of evidence (0-1)

    Returns:
        Overall confidence score between 0 and 1

    Raises:
        ValueError: If verdicts and scores have different lengths

    Examples:
        >>> calculate_confidence(
        ...     ["SUPPORTED", "SUPPORTED"],
        ...     [0.9, 0.8]
        ... )
        0.85
    """
    if len(verdicts) != len(evidence_scores):
        raise ValueError("Verdicts and scores must have same length")

    return sum(evidence_scores) / len(evidence_scores)
```

### Code Formatting

```bash
# Format code with Black
black detect/ tests/

# Sort imports with isort
isort detect/ tests/

# Check types with mypy
mypy detect/

# Lint with flake8
flake8 detect/ tests/

# All-in-one command
make format  # Runs black, isort, flake8, mypy
```

**Pre-commit configuration** (.pre-commit-config.yaml):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,W503']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## Development Workflow

### Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/add-multilingual-support

# 2. Make changes
vim detect/agents/classifier.py

# 3. Run tests
pytest tests/agents/test_classifier.py

# 4. Format and lint
make format

# 5. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat(classifier): add multilingual topic classification

- Add support for French, Spanish, German
- Update NER models for multi-language
- Add language detection pre-processing
"

# 6. Push branch
git push origin feature/add-multilingual-support

# 7. Create Pull Request on GitHub
```

### Commit Message Convention

Follow **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(api): add batch processing endpoint

- Implement /batch-check endpoint
- Add queue management with RabbitMQ
- Support CSV and JSON input formats

Closes #123

fix(fact-checker): handle timeout errors gracefully

Previously, LLM timeouts crashed the entire pipeline.
Now we retry with exponential backoff and fall back
to cached results if available.

Fixes #456
```

---

## Adding Features

### Example: Adding a New Agent

**Step 1: Create Agent Class**

```python
# detect/agents/sentiment.py

from detect.agents.base import BaseAgent
from detect.types import FactCheckingState
from transformers import pipeline

class SentimentAgent(BaseAgent):
    """Analyzes sentiment of claims to detect bias"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )

    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Analyze sentiment of claim and evidence"""
        claim = state['original_claim']

        # Analyze claim sentiment
        claim_sentiment = self.analyzer(claim)[0]
        state['claim_sentiment'] = {
            'label': claim_sentiment['label'],
            'score': claim_sentiment['score']
        }

        # Analyze evidence sentiment
        if 'evidence_retrieved' in state:
            evidence_sentiments = []
            for evidence in state['evidence_retrieved'][:5]:
                sent = self.analyzer(evidence['text'][:512])[0]
                evidence_sentiments.append(sent['label'])

            state['evidence_sentiment_distribution'] = {
                'positive': evidence_sentiments.count('POSITIVE'),
                'negative': evidence_sentiments.count('NEGATIVE'),
                'neutral': evidence_sentiments.count('NEUTRAL')
            }

        self.update_trace(
            state,
            f"Sentiment: {claim_sentiment['label']} ({claim_sentiment['score']:.2f})"
        )

        return state
```

**Step 2: Add Tests**

```python
# tests/agents/test_sentiment.py

import pytest
from detect.agents.sentiment import SentimentAgent
from detect.types import FactCheckingState

@pytest.fixture
def sentiment_agent():
    config = {"model": "cardiffnlp/twitter-roberta-base-sentiment"}
    return SentimentAgent(config)

def test_sentiment_analysis(sentiment_agent):
    """Test basic sentiment analysis"""
    state = FactCheckingState(
        original_claim="This is amazing and wonderful news!",
        reasoning_trace=[],
        agents_involved=[]
    )

    result = sentiment_agent.process(state)

    assert 'claim_sentiment' in result
    assert result['claim_sentiment']['label'] == 'POSITIVE'
    assert result['claim_sentiment']['score'] > 0.5

def test_evidence_sentiment_distribution(sentiment_agent):
    """Test evidence sentiment analysis"""
    state = FactCheckingState(
        original_claim="Test claim",
        evidence_retrieved=[
            {'text': 'This is great!'},
            {'text': 'This is terrible.'},
            {'text': 'Neutral statement.'}
        ],
        reasoning_trace=[],
        agents_involved=[]
    )

    result = sentiment_agent.process(state)

    assert 'evidence_sentiment_distribution' in result
    dist = result['evidence_sentiment_distribution']
    assert 'positive' in dist
    assert 'negative' in dist
```

**Step 3: Integrate into Workflow**

```python
# detect/orchestration/langgraph.py

from detect.agents.sentiment import SentimentAgent

# Add to workflow
workflow.add_node("sentiment", SentimentAgent(config))

# Add edge after classifier
workflow.add_edge("classifier", "sentiment")
workflow.add_edge("sentiment", "anomaly_detector")
```

**Step 4: Update Documentation**

Update `docs/AGENTS.md` with new agent documentation.

---

## Debugging

### Using pdb/ipdb

```python
# Insert breakpoint
import ipdb; ipdb.set_trace()

# Or using Python 3.7+ built-in
breakpoint()
```

### Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

# Log levels
logger.debug("Detailed diagnostic info")
logger.info("General informational messages")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical errors")

# Structured logging
logger.info(
    "Fact-check completed",
    extra={
        "claim_id": claim_id,
        "verdict": verdict,
        "confidence": confidence,
        "duration_ms": duration
    }
)
```

### Debugging Agents

```python
# Enable verbose logging
import os
os.environ['LOG_LEVEL'] = 'DEBUG'

# Test agent in isolation
from detect.agents.fact_checker import FactCheckerAgent
from detect.types import FactCheckingState

agent = FactCheckerAgent(config)
state = FactCheckingState(
    original_claim="Test claim",
    decomposed_assertions=["Assertion 1"],
    # ... other fields
)

result = agent.process(state)
print(result['triplet_verdicts'])
```

### Using pytest for interactive debugging

```bash
# Drop into debugger on failure
pytest tests/agents/test_fact_checker.py --pdb

# Drop into debugger on error
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb --pdb

# Run specific test with verbose output
pytest tests/agents/test_fact_checker.py::test_rag_verification -vvs
```

---

## Performance Optimization

### Profiling

```bash
# Profile with py-spy
py-spy record -o profile.svg -- python -m detect.main

# Memory profiling
mprof run python -m detect.main
mprof plot
```

### Caching Strategies

```python
from functools import lru_cache
import redis

# In-memory cache
@lru_cache(maxsize=1000)
def get_source_credibility(domain: str) -> float:
    """Cache credibility scores"""
    return calculate_credibility(domain)

# Redis cache decorator
def redis_cache(ttl: int = 3600):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            cached = redis_client.get(cache_key)

            if cached:
                return json.loads(cached)

            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result

        return wrapper
    return decorator

@redis_cache(ttl=3600)
def retrieve_evidence(query: str):
    """Expensive retrieval operation"""
    return expensive_search(query)
```

### Async/Await for I/O

```python
import asyncio
import httpx

async def fetch_multiple_sources(urls: List[str]) -> List[str]:
    """Fetch multiple URLs in parallel"""
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.text for r in responses]

# Usage
results = asyncio.run(fetch_multiple_sources(url_list))
```

---

## Best Practices

### Error Handling

```python
from detect.exceptions import (
    ClaimValidationError,
    EvidenceRetrievalError,
    LLMServiceError
)

def process_claim(claim: str) -> Result:
    """Process claim with proper error handling"""
    try:
        # Validate input
        if not claim or len(claim) < 10:
            raise ClaimValidationError(
                "Claim must be at least 10 characters",
                claim=claim
            )

        # Retrieve evidence
        try:
            evidence = retrieve_evidence(claim)
        except TimeoutError:
            raise EvidenceRetrievalError(
                "Evidence retrieval timed out",
                claim=claim,
                timeout=30
            )

        # Verify with LLM
        try:
            verdict = llm_verify(claim, evidence)
        except Exception as e:
            raise LLMServiceError(
                f"LLM service failed: {str(e)}",
                claim=claim,
                cause=e
            )

        return Result(verdict=verdict)

    except ClaimValidationError:
        logger.warning(f"Invalid claim: {claim}")
        raise

    except (EvidenceRetrievalError, LLMServiceError) as e:
        logger.error(f"Processing failed: {e}")
        # Graceful degradation
        return Result(
            verdict="INSUFFICIENT_INFO",
            confidence=0.0,
            error=str(e)
        )
```

### Configuration Management

```python
# Use pydantic for configuration
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Database
    database_url: str
    redis_url: str

    # LLM
    anthropic_api_key: str
    openai_api_key: str

    # Feature flags
    enable_deepfake_detection: bool = True
    enable_graph_reasoning: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

### Testing Strategies

```python
# Use fixtures for common setup
@pytest.fixture
def mock_llm_service(mocker):
    """Mock LLM service for testing"""
    mock = mocker.Mock()
    mock.generate.return_value = json.dumps({
        "verdict": "SUPPORTED",
        "confidence": 0.9,
        "reasoning": "Evidence supports claim"
    })
    return mock

def test_fact_checking_with_mock_llm(mock_llm_service):
    """Test fact-checking with mocked LLM"""
    agent = FactCheckerAgent(config)
    agent.llm = mock_llm_service

    result = agent.process(test_state)

    assert result['final_verdict'] == "SUPPORTED"
    assert mock_llm_service.generate.called
```

---

## Makefile

```makefile
# Makefile for common tasks

.PHONY: help install test format lint clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make format     - Format code"
	@echo "  make lint       - Lint code"
	@echo "  make clean      - Clean build artifacts"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=detect --cov-report=html

test-watch:
	pytest-watch tests/ -v

format:
	black detect/ tests/
	isort detect/ tests/

lint:
	flake8 detect/ tests/
	mypy detect/
	pylint detect/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf build/ dist/ *.egg-info

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-logs:
	docker-compose logs -f

run-dev:
	uvicorn detect.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Next Steps

- Read [TESTING.md](TESTING.md) for testing guidelines
- Review [SECURITY.md](SECURITY.md) for security best practices
- Check [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs

---

**Last Updated**: 2025-01-15
