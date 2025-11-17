# Testing Guide

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Unit Tests](#unit-tests)
3. [Integration Tests](#integration-tests)
4. [End-to-End Tests](#end-to-end-tests)
5. [Test Coverage](#test-coverage)
6. [Performance Testing](#performance-testing)
7. [Testing Best Practices](#testing-best-practices)

---

## Testing Strategy

### Test Pyramid

```
          /\
         /  \    E2E Tests (5%)
        /────\   - Full workflow tests
       /      \  - User acceptance tests
      /────────\
     /          \ Integration Tests (25%)
    /────────────\ - Agent interactions
   /              \ - Database operations
  /────────────────\ - API endpoint tests
 /                  \
/____________________ Unit Tests (70%)
  - Agent methods
  - Utils functions
  - Service logic
```

### Test Targets

| Metric | Target | Current |
|--------|--------|---------|
| Overall Coverage | > 80% | 78% |
| Critical Paths | 100% | 95% |
| Agent Logic | > 90% | 92% |
| API Endpoints | 100% | 100% |

---

## Unit Tests

### Agent Tests

```python
# tests/unit/agents/test_classifier.py

import pytest
from detect.agents.classifier import ClassifierAgent
from detect.types import FactCheckingState

@pytest.fixture
def classifier_config():
    """Classifier configuration for tests"""
    return {
        "nlp_model": "en_core_web_sm",
        "topic_classifier": "facebook/bart-large-mnli"
    }

@pytest.fixture
def classifier_agent(classifier_config, mocker):
    """Create classifier agent with mocked dependencies"""
    # Mock LLM service
    mocker.patch(
        'detect.agents.classifier.LLMService'
    )

    return ClassifierAgent(classifier_config)

class TestClassifierAgent:
    """Test suite for Classifier Agent"""

    def test_entity_extraction(self, classifier_agent):
        """Test NER entity extraction"""
        state = FactCheckingState(
            original_claim="Jean Dupont, CEO of TechCorp, announced 150% growth",
            reasoning_trace=[],
            agents_involved=[]
        )

        result = classifier_agent.process(state)

        assert 'entities' in result
        assert len(result['entities']['PERSON']) > 0
        assert any('Jean Dupont' in p['text'] for p in result['entities']['PERSON'])
        assert len(result['entities']['ORG']) > 0
        assert len(result['entities']['PERCENT']) > 0

    def test_topic_classification(self, classifier_agent):
        """Test topic classification accuracy"""
        state = FactCheckingState(
            original_claim="The new COVID-19 variant spreads faster than previous strains",
            reasoning_trace=[],
            agents_involved=[]
        )

        result = classifier_agent.process(state)

        assert 'classification' in result
        assert result['classification']['theme'] == 'health'
        assert result['classification']['all_topics'][0]['score'] > 0.5

    def test_claim_decomposition(self, classifier_agent, mocker):
        """Test claim decomposition into assertions"""
        mock_llm = mocker.patch.object(
            classifier_agent,
            'llm'
        )
        mock_llm.generate.return_value = json.dumps([
            "Assertion 1",
            "Assertion 2",
            "Assertion 3"
        ])

        state = FactCheckingState(
            original_claim="Complex claim with multiple facts",
            reasoning_trace=[],
            agents_involved=[]
        )

        result = classifier_agent.process(state)

        assert 'decomposed_assertions' in result
        assert len(result['decomposed_assertions']) == 3
        assert mock_llm.generate.called

    @pytest.mark.parametrize("claim,expected_complexity", [
        ("Simple claim", 2),  # Simple
        ("Complex claim with Jean Dupont, TechCorp, 150% growth, and statistics", 8),  # Complex
        ("Very long claim " * 20, 6),  # Long claim
    ])
    def test_complexity_calculation(self, classifier_agent, claim, expected_complexity):
        """Test complexity score calculation"""
        result = classifier_agent._calculate_complexity(
            claim,
            assertions=["Assertion 1", "Assertion 2"],
            entities={'PERSON': [{'text': 'Jean'}], 'ORG': []}
        )

        assert isinstance(result, int)
        assert 1 <= result <= 10
        assert abs(result - expected_complexity) <= 2  # Allow some variance
```

### Service Tests

```python
# tests/unit/services/test_retrieval.py

import pytest
from detect.services.retrieval import HybridRetriever

@pytest.fixture
def retriever(mocker):
    """Create retriever with mocked dependencies"""
    config = {
        "bm25": {"k1": 1.5, "b": 0.75},
        "semantic": {"model": "all-MiniLM-L6-v2"}
    }

    # Mock vector store
    mocker.patch('detect.services.retrieval.Weaviate')

    return HybridRetriever(config)

class TestHybridRetriever:
    """Test suite for Hybrid Retrieval"""

    def test_bm25_retrieval(self, retriever):
        """Test BM25 sparse retrieval"""
        documents = retriever.bm25.get_relevant_documents(
            "Jean Dupont TechCorp",
            k=5
        )

        assert len(documents) <= 5
        assert all(hasattr(doc, 'page_content') for doc in documents)

    def test_semantic_retrieval(self, retriever):
        """Test semantic dense retrieval"""
        documents = retriever.semantic.get_relevant_documents(
            "Who is the CEO of TechCorp?",
            k=5
        )

        assert len(documents) <= 5
        assert all(hasattr(doc, 'metadata') for doc in documents)

    def test_ensemble_retrieval(self, retriever, mocker):
        """Test ensemble retrieval combines both methods"""
        mock_bm25_docs = [mocker.Mock(page_content=f"BM25 doc {i}") for i in range(5)]
        mock_semantic_docs = [mocker.Mock(page_content=f"Semantic doc {i}") for i in range(5)]

        mocker.patch.object(
            retriever.bm25,
            'get_relevant_documents',
            return_value=mock_bm25_docs
        )
        mocker.patch.object(
            retriever.semantic,
            'get_relevant_documents',
            return_value=mock_semantic_docs
        )

        documents = retriever.retrieve("test query", k=10)

        assert len(documents) == 10
        retriever.bm25.get_relevant_documents.assert_called_once()
        retriever.semantic.get_relevant_documents.assert_called_once()
```

---

## Integration Tests

### Agent Workflow Tests

```python
# tests/integration/test_workflow.py

import pytest
from detect.orchestration.langgraph import build_workflow
from detect.types import FactCheckingState

@pytest.mark.integration
class TestAgentWorkflow:
    """Test agent workflow integration"""

    @pytest.fixture
    def workflow(self):
        """Build complete workflow"""
        config = load_test_config()
        return build_workflow(config)

    def test_full_pipeline_supported_claim(self, workflow):
        """Test full pipeline with supported claim"""
        initial_state = FactCheckingState(
            original_claim="Paris is the capital of France",
            decomposed_assertions=[],
            reasoning_trace=[],
            agents_involved=[],
            created_at=datetime.now().isoformat()
        )

        result = workflow.invoke(initial_state)

        # Verify all agents executed
        assert len(result['agents_involved']) == 5
        assert 'CollectorAgent' in result['agents_involved']
        assert 'ClassifierAgent' in result['agents_involved']
        assert 'AnomalyDetectorAgent' in result['agents_involved']
        assert 'FactCheckerAgent' in result['agents_involved']
        assert 'ReporterAgent' in result['agents_involved']

        # Verify final result
        assert result['final_verdict'] == 'SUPPORTED'
        assert result['confidence'] > 0.8
        assert len(result['evidence_retrieved']) > 0

    def test_full_pipeline_refuted_claim(self, workflow):
        """Test full pipeline with refuted claim"""
        initial_state = FactCheckingState(
            original_claim="The Earth is flat",
            decomposed_assertions=[],
            reasoning_trace=[],
            agents_involved=[],
            created_at=datetime.now().isoformat()
        )

        result = workflow.invoke(initial_state)

        assert result['final_verdict'] == 'REFUTED'
        assert result['confidence'] > 0.7
        assert len(result['evidence_retrieved']) > 0

    def test_complex_claim_uses_graph_reasoning(self, workflow):
        """Test that complex claims trigger graph-based reasoning"""
        complex_claim = (
            "Jean Dupont, who became CEO of TechCorp in 2020 after "
            "serving as CFO for 5 years, announced that the company's "
            "revenue grew by 150% in Q4 2024, surpassing analyst "
            "expectations of 100% growth."
        )

        initial_state = FactCheckingState(
            original_claim=complex_claim,
            decomposed_assertions=[],
            reasoning_trace=[],
            agents_involved=[],
            created_at=datetime.now().isoformat()
        )

        result = workflow.invoke(initial_state)

        # Complex claims should have complexity > 7
        assert result['classification']['complexity'] > 7

        # Should have multiple assertions
        assert len(result['decomposed_assertions']) >= 4
```

### Database Integration Tests

```python
# tests/integration/test_database.py

import pytest
from detect.db.postgres import FactCheckDatabase
from detect.models.claim import FactCheck

@pytest.mark.integration
class TestDatabaseOperations:
    """Test database operations"""

    @pytest.fixture
    def db(self):
        """Create test database connection"""
        db = FactCheckDatabase(DATABASE_TEST_URL)
        db.create_tables()
        yield db
        db.drop_tables()

    def test_save_and_retrieve_fact_check(self, db):
        """Test saving and retrieving fact-check"""
        fact_check = FactCheck(
            claim="Test claim",
            verdict="SUPPORTED",
            confidence=0.87,
            reasoning_trace=["Step 1", "Step 2"],
            evidence_used=[{"source": "test.com", "text": "Evidence"}]
        )

        # Save
        claim_id = db.save_fact_check(fact_check)
        assert claim_id is not None

        # Retrieve
        retrieved = db.get_fact_check(claim_id)
        assert retrieved is not None
        assert retrieved.claim == "Test claim"
        assert retrieved.verdict == "SUPPORTED"
        assert retrieved.confidence == 0.87

    def test_search_similar_claims(self, db):
        """Test finding similar claims"""
        # Save multiple claims
        db.save_fact_check(FactCheck(
            claim="Paris is the capital of France",
            verdict="SUPPORTED",
            confidence=0.99
        ))
        db.save_fact_check(FactCheck(
            claim="Paris is the largest city in France",
            verdict="SUPPORTED",
            confidence=0.95
        ))

        # Search for similar
        similar = db.search_similar_claims(
            "What is the capital of France?",
            threshold=0.7,
            limit=5
        )

        assert len(similar) >= 1
        assert any('Paris' in claim.claim for claim in similar)
```

---

## End-to-End Tests

### API E2E Tests

```python
# tests/e2e/test_api.py

import pytest
from fastapi.testclient import TestClient
from detect.main import app

@pytest.mark.e2e
class TestAPIEndpoints:
    """End-to-end API tests"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def api_key(self):
        """Test API key"""
        return "test_api_key_123"

    def test_fact_check_endpoint_full_flow(self, client, api_key):
        """Test complete fact-checking flow via API"""
        response = client.post(
            "/v1/fact-check",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "claim": "Paris is the capital of France",
                "options": {
                    "include_evidence": True,
                    "include_reasoning": True
                }
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert 'claim_id' in data
        assert 'verdict' in data
        assert 'confidence' in data
        assert 'evidence' in data
        assert 'reasoning_trace' in data

        # Verify verdict
        assert data['verdict'] in ['SUPPORTED', 'REFUTED', 'INSUFFICIENT_INFO']
        assert 0 <= data['confidence'] <= 1

        # Verify evidence
        assert len(data['evidence']) > 0
        assert all('source' in e for e in data['evidence'])
        assert all('credibility' in e for e in data['evidence'])

    def test_batch_processing_endpoint(self, client, api_key):
        """Test batch processing endpoint"""
        # Submit batch
        response = client.post(
            "/v1/batch-check",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "claims": [
                    {"id": "1", "claim": "Paris is the capital of France"},
                    {"id": "2", "claim": "The Earth is flat"}
                ]
            }
        )

        assert response.status_code == 202
        batch_data = response.json()
        batch_id = batch_data['batch_id']

        # Poll for completion
        import time
        for _ in range(30):  # Max 30 seconds
            status_response = client.get(
                f"/v1/batch-check/{batch_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )

            if status_response.json()['status'] == 'completed':
                break

            time.sleep(1)

        # Verify results
        results = status_response.json()
        assert results['status'] == 'completed'
        assert len(results['results']) == 2
        assert results['results'][0]['verdict'] == 'SUPPORTED'
        assert results['results'][1]['verdict'] == 'REFUTED'

    def test_rate_limiting(self, client, api_key):
        """Test rate limiting enforcement"""
        # Make requests until rate limited
        responses = []
        for _ in range(150):  # Exceed limit
            response = client.post(
                "/v1/fact-check",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"claim": "Test claim"}
            )
            responses.append(response.status_code)

        # Should have some 429 responses
        assert 429 in responses

    def test_unauthorized_access(self, client):
        """Test authentication requirement"""
        response = client.post(
            "/v1/fact-check",
            json={"claim": "Test claim"}
        )

        assert response.status_code == 401
```

---

## Test Coverage

### Measuring Coverage

```bash
# Run tests with coverage
pytest --cov=detect --cov-report=html --cov-report=term

# Generate coverage badge
coverage-badge -o coverage.svg

# View HTML report
open htmlcov/index.html
```

### Coverage Configuration

```ini
# .coveragerc

[run]
source = detect
omit =
    */tests/*
    */venv/*
    */__pycache__/*
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:

[html]
directory = htmlcov
```

---

## Performance Testing

### Load Testing

```python
# tests/performance/test_load.py

import pytest
from locust import HttpUser, task, between

class FactCheckUser(HttpUser):
    """Simulated user for load testing"""

    wait_time = between(1, 3)

    @task(3)
    def check_simple_claim(self):
        """Test simple fact-check"""
        self.client.post(
            "/v1/fact-check",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"claim": "Paris is the capital of France"}
        )

    @task(1)
    def check_complex_claim(self):
        """Test complex fact-check"""
        self.client.post(
            "/v1/fact-check",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "claim": "Jean Dupont, CEO of TechCorp since 2020, announced 150% revenue growth",
                "options": {"include_evidence": True}
            }
        )

# Run with: locust -f tests/performance/test_load.py --host=http://localhost:8000
```

### Benchmark Tests

```python
# tests/performance/test_benchmarks.py

import pytest
import time
from detect.agents.fact_checker import FactCheckerAgent

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def test_fact_checker_latency(self, benchmark):
        """Benchmark fact-checker latency"""
        agent = FactCheckerAgent(config)
        state = create_test_state()

        result = benchmark(agent.process, state)

        assert result is not None
        # Should complete in < 30 seconds
        assert benchmark.stats.mean < 30.0

    def test_retrieval_throughput(self, benchmark):
        """Benchmark retrieval throughput"""
        retriever = HybridRetriever(config)

        def retrieve_batch():
            queries = [f"Query {i}" for i in range(10)]
            return [retriever.retrieve(q) for q in queries]

        result = benchmark(retrieve_batch)

        # Should process 10 queries in < 5 seconds
        assert benchmark.stats.mean < 5.0
```

---

## Testing Best Practices

### 1. Use Fixtures

```python
@pytest.fixture(scope="session")
def test_db():
    """Session-scoped database"""
    db = create_test_database()
    yield db
    db.cleanup()

@pytest.fixture
def mock_llm(mocker):
    """Mock LLM for all tests"""
    return mocker.patch('detect.services.llm.LLMService')
```

### 2. Parametrize Tests

```python
@pytest.mark.parametrize("claim,expected_verdict", [
    ("Paris is the capital of France", "SUPPORTED"),
    ("The Earth is flat", "REFUTED"),
    ("Unknown fact", "INSUFFICIENT_INFO"),
])
def test_verdicts(claim, expected_verdict):
    result = check_claim(claim)
    assert result.verdict == expected_verdict
```

### 3. Use Factories

```python
import factory

class FactCheckFactory(factory.Factory):
    class Meta:
        model = FactCheck

    claim = factory.Faker('sentence')
    verdict = factory.Iterator(['SUPPORTED', 'REFUTED', 'INSUFFICIENT_INFO'])
    confidence = factory.Faker('pyfloat', min_value=0, max_value=1)

# Usage
fact_check = FactCheckFactory()
```

### 4. Separate Test Types

```python
# Mark test types
@pytest.mark.unit
def test_unit():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.e2e
def test_e2e():
    pass

# Run specific type
# pytest -m unit
# pytest -m "not e2e"
```

---

**Target**: 80%+ test coverage, < 30s average latency, 0 critical bugs in production

**Last Updated**: 2025-01-15
