# System Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Agent System](#agent-system)
6. [Orchestration Patterns](#orchestration-patterns)
7. [State Management](#state-management)
8. [Integration Points](#integration-points)
9. [Scalability & Performance](#scalability--performance)
10. [Design Decisions](#design-decisions)

---

## Overview

The Multi-Agent Disinformation Detection System is built on a **microservices-oriented, agent-based architecture** that enables modular, scalable, and maintainable fact-checking capabilities.

### Core Principles

- **Modularity**: Each agent is independent and replaceable
- **Scalability**: Horizontal scaling at agent and service levels
- **Observability**: Comprehensive logging and tracing
- **Resilience**: Fault-tolerant with graceful degradation
- **Extensibility**: Easy to add new agents or data sources

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Streamlit  │  │   REST API   │  │   GraphQL    │  │   WebSocket  │   │
│  │   Dashboard  │  │   (FastAPI)  │  │   Gateway    │  │   (Real-time)│   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                  │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                      LangGraph Workflow Engine                     │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │     │
│  │  │   Routing    │  │  State Mgmt  │  │  Pattern Execution   │    │     │
│  │  │   Logic      │  │  & Memory    │  │  (Sequential/Parallel)│   │     │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────────────────┐
│                            AGENT LAYER                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Agent 1:   │  │  Agent 2:   │  │  Agent 3:   │  │  Agent 4:   │       │
│  │  Collector  │→ │ Classifier  │→ │  Anomaly    │→ │Fact-Checker │       │
│  │  & Indexer  │  │             │  │  Detector   │  │             │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│         ↓                                                    ↓               │
│  ┌─────────────┐                                   ┌─────────────┐         │
│  │  Agent 5:   │                                   │  Deepfake   │         │
│  │  Reporter & │                                   │  Detection  │         │
│  │  Alerter    │                                   │  Module     │         │
│  └─────────────┘                                   └─────────────┘         │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────────────────┐
│                          SERVICE LAYER                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Hybrid    │  │   LLM       │  │  Embedding  │  │   Source    │       │
│  │  Retrieval  │  │  Service    │  │  Service    │  │  Credibility│       │
│  │  (RAG)      │  │  (Claude)   │  │ (HuggingFace)│ │  Scorer     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Graph     │  │  Web Search │  │  NER/NLP    │  │  Media      │       │
│  │  Reasoning  │  │  Service    │  │  Service    │  │  Analysis   │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ PostgreSQL  │  │   Redis     │  │  Weaviate   │  │  RabbitMQ   │       │
│  │ (Primary)   │  │  (Cache)    │  │ (Vectors)   │  │  (Queue)    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │    S3       │  │  Prometheus │  │  Elasticsearch│                       │
│  │ (Objects)   │  │  (Metrics)  │  │   (Logs)    │                        │
│  └─────────────┘  └─────────────┘  └─────────────┘                        │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────────────────┐
│                         EXTERNAL SOURCES                                     │
│  Twitter/X │ Reddit │ Google Trends │ NewsAPI │ Fact-Check DBs │ RSS Feeds  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Presentation Layer

#### REST API (FastAPI)
- **Endpoints**: `/fact-check`, `/batch-process`, `/status`, `/history`
- **Authentication**: JWT-based with API keys
- **Rate Limiting**: 100 requests/minute per user
- **Documentation**: Auto-generated OpenAPI/Swagger

```python
# API Structure
app/
├── api/
│   ├── v1/
│   │   ├── fact_check.py      # Main fact-checking endpoints
│   │   ├── batch.py            # Batch processing
│   │   ├── monitoring.py       # Health & metrics
│   │   └── webhooks.py         # Webhook integrations
│   └── middleware/
│       ├── auth.py             # JWT verification
│       ├── rate_limit.py       # Rate limiting
│       └── cors.py             # CORS handling
```

#### Dashboard (Streamlit)
- Real-time monitoring of fact-checking pipeline
- Interactive claim submission and analysis
- Historical data visualization
- Alert management interface

### 2. Orchestration Layer

#### LangGraph Workflow Engine

The orchestration engine manages agent execution using **LangGraph's StateGraph**:

```python
from langgraph.graph import StateGraph, END

class FactCheckingState(TypedDict):
    """Shared state across all agents"""
    original_claim: str
    decomposed_assertions: List[str]
    classification: Dict
    evidence_retrieved: List[Dict]
    anomaly_scores: Dict
    triplet_verdicts: Dict
    final_verdict: str
    confidence: float
    reasoning_trace: List[str]
    created_at: str
    agents_involved: List[str]

# Build workflow
workflow = StateGraph(FactCheckingState)
workflow.add_node("collector", agent_collector)
workflow.add_node("classifier", agent_classifier)
workflow.add_node("anomaly_detector", agent_anomaly_detector)
workflow.add_node("fact_checker", agent_fact_checker)
workflow.add_node("reporter", agent_reporter)

# Sequential execution
workflow.add_edge("START", "collector")
workflow.add_edge("collector", "classifier")
workflow.add_edge("classifier", "anomaly_detector")
workflow.add_edge("anomaly_detector", "fact_checker")
workflow.add_edge("fact_checker", "reporter")
workflow.add_edge("reporter", END)

app = workflow.compile()
```

**Key Features**:
- **Conditional Routing**: Dynamic branching based on claim complexity
- **Parallel Execution**: Multiple evidence retrievals in parallel
- **Checkpointing**: Resume from failure points
- **Tracing**: Complete execution history

### 3. Agent Layer

Each agent is a self-contained module with:

1. **Input Specification**: What state it requires
2. **Processing Logic**: Core functionality
3. **Output Specification**: What state it modifies
4. **Error Handling**: Graceful degradation strategies

**Agent Interface**:
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        self.metrics = MetricsCollector()

    @abstractmethod
    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Main processing logic"""
        pass

    def pre_process(self, state: FactCheckingState) -> None:
        """Pre-processing hooks"""
        self.logger.info(f"Starting {self.__class__.__name__}")
        self.metrics.increment("agent_invocations")

    def post_process(self, state: FactCheckingState) -> None:
        """Post-processing hooks"""
        self.logger.info(f"Completed {self.__class__.__name__}")
        state['agents_involved'].append(self.__class__.__name__)
```

### 4. Service Layer

#### Hybrid Retrieval Service (RAG)

Combines **BM25** (sparse) and **semantic search** (dense):

```python
class HybridRetriever:
    def __init__(self):
        # Sparse retriever
        self.bm25 = BM25Retriever(k1=1.5, b=0.75)

        # Dense retriever
        self.embeddings = HuggingFaceEmbeddings("all-MiniLM-L6-v2")
        self.vector_store = Weaviate(embeddings=self.embeddings)

        # Ensemble
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.vector_store],
            weights=[0.4, 0.6]  # Favor semantic
        )

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Hybrid retrieval with re-ranking"""
        # Initial retrieval
        docs = self.ensemble.get_relevant_documents(query, k=k*2)

        # Re-rank with cross-encoder
        reranked = self.rerank(query, docs)

        return reranked[:k]
```

#### LLM Service

Abstraction layer supporting multiple LLM providers:

```python
class LLMService:
    """Unified LLM interface"""

    def __init__(self, provider: str = "claude"):
        self.provider = self._init_provider(provider)
        self.cache = RedisCache()

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        use_cache: bool = True
    ) -> str:
        """Generate response with caching"""

        # Check cache
        if use_cache:
            cache_key = self._hash_prompt(prompt)
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        # Generate
        response = await self.provider.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Cache result
        if use_cache:
            await self.cache.set(cache_key, response, ttl=3600)

        return response
```

### 5. Data Layer

#### PostgreSQL Schema

```sql
-- Fact check logs
CREATE TABLE fact_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim TEXT NOT NULL,
    verdict VARCHAR(50) NOT NULL,  -- SUPPORTED/REFUTED/INSUFFICIENT_INFO
    confidence FLOAT NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    reasoning_trace JSONB NOT NULL,
    evidence_used JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    agents_involved JSONB NOT NULL,
    metadata JSONB
);

-- Evidence sources
CREATE TABLE evidence_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT UNIQUE NOT NULL,
    domain VARCHAR(255) NOT NULL,
    credibility_score FLOAT CHECK (credibility_score BETWEEN 0 AND 1),
    last_verified TIMESTAMP,
    metadata JSONB
);

-- Claim assertions (for decomposition)
CREATE TABLE claim_assertions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_check_id UUID REFERENCES fact_checks(id) ON DELETE CASCADE,
    assertion TEXT NOT NULL,
    verdict VARCHAR(50),
    confidence FLOAT,
    evidence JSONB
);

-- Alerts
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_check_id UUID REFERENCES fact_checks(id),
    severity VARCHAR(20) NOT NULL,  -- LOW/MEDIUM/HIGH/CRITICAL
    message TEXT NOT NULL,
    sent_at TIMESTAMP DEFAULT NOW(),
    recipients JSONB
);

-- Indexes for performance
CREATE INDEX idx_fact_checks_created_at ON fact_checks(created_at DESC);
CREATE INDEX idx_fact_checks_verdict ON fact_checks(verdict);
CREATE INDEX idx_evidence_sources_domain ON evidence_sources(domain);
CREATE INDEX idx_evidence_sources_credibility ON evidence_sources(credibility_score DESC);
```

#### Redis Cache Strategy

```python
# Cache patterns
CACHE_PATTERNS = {
    # LLM responses (1 hour)
    "llm:response:{prompt_hash}": 3600,

    # Evidence retrieval (30 minutes)
    "evidence:{query_hash}": 1800,

    # Source credibility (24 hours)
    "credibility:{domain}": 86400,

    # Embeddings (1 week)
    "embedding:{text_hash}": 604800,

    # API rate limits (1 minute)
    "ratelimit:{user_id}": 60
}
```

---

## Data Flow

### End-to-End Fact-Checking Flow

```
1. USER SUBMITS CLAIM
   │
   ├─→ POST /api/v1/fact-check
   │   Body: {"claim": "Jean Dupont, CEO of TechCorp..."}
   │
   ▼
2. API GATEWAY
   │
   ├─→ Authenticate (JWT)
   ├─→ Rate Limit Check
   ├─→ Input Validation
   │
   ▼
3. ORCHESTRATOR CREATES INITIAL STATE
   │
   state = {
       "original_claim": "...",
       "decomposed_assertions": [],
       "classification": {},
       ...
   }
   │
   ▼
4. AGENT 1: COLLECTOR
   │
   ├─→ Normalize claim text
   ├─→ Check for duplicates
   ├─→ Assign unique ID
   │
   ▼
5. AGENT 2: CLASSIFIER
   │
   ├─→ NER extraction (entities)
   ├─→ Topic classification (politics/health/etc)
   ├─→ Complexity scoring (1-10)
   ├─→ Decompose into assertions
   │   state["decomposed_assertions"] = [assertion1, assertion2, ...]
   │
   ▼
6. AGENT 3: ANOMALY DETECTOR
   │
   ├─→ For each assertion:
   │   ├─→ Semantic coherence check (LLM)
   │   ├─→ Pattern matching (known disinfo patterns)
   │   ├─→ Tone analysis (manipulative language)
   │   └─→ Assign suspicion score (0-1)
   │
   │   state["anomaly_scores"] = {assertion: score}
   │
   ▼
7. AGENT 4: FACT-CHECKER
   │
   ├─→ DECISION: If complexity > 7 → Graph-based reasoning
   │             Else → Standard RAG
   │
   ├─→ RETRIEVAL (Parallel):
   │   ├─→ Hybrid search (BM25 + Semantic)
   │   ├─→ Web search (DuckDuckGo/Google)
   │   ├─→ Fact-check DB query (Snopes, PolitiFact)
   │   └─→ Knowledge base lookup
   │
   ├─→ EVIDENCE SCORING:
   │   ├─→ Source credibility (0-1)
   │   ├─→ Relevance to assertion (0-1)
   │   └─→ Recency weight
   │
   ├─→ VERIFICATION (LLM Chain-of-Thought):
   │   For each assertion:
   │     1. Decompose into sub-claims
   │     2. Match with evidence
   │     3. Apply logical reasoning
   │     4. Generate verdict + confidence
   │
   │   state["triplet_verdicts"] = {assertion: {verdict, confidence}}
   │
   ▼
8. AGENT 5: REPORTER
   │
   ├─→ CONSOLIDATION:
   │   ├─→ Merge assertion verdicts
   │   ├─→ Calculate final confidence
   │   └─→ Final verdict = f(assertion_verdicts)
   │
   ├─→ GENERATE REPORT:
   │   ├─→ Reasoning trace (full transparency)
   │   ├─→ Evidence summary
   │   ├─→ Confidence intervals
   │   └─→ Recommendations
   │
   ├─→ ALERT DECISION:
   │   If (verdict == REFUTED AND confidence > 0.8 AND virality_score > threshold):
   │       → Send alert to journalists/regulators
   │
   ▼
9. PERSIST RESULTS
   │
   ├─→ Save to PostgreSQL (fact_checks table)
   ├─→ Update evidence sources credibility
   ├─→ Log metrics to Prometheus
   ├─→ Index in Elasticsearch
   │
   ▼
10. RETURN RESPONSE
    │
    {
      "claim_id": "uuid",
      "verdict": "REFUTED",
      "confidence": 0.87,
      "reasoning_trace": [...],
      "evidence": [...],
      "processing_time": 24.5
    }
```

---

## Agent System

### Agent Communication Patterns

#### 1. Sequential Pipeline (Default)
```
Collector → Classifier → Anomaly Detector → Fact-Checker → Reporter
```

#### 2. Parallel Processing (Evidence Retrieval)
```
                    ┌─→ Web Search ─────┐
Fact-Checker ───────┼─→ Fact-Check DB ──┤──→ Merge Results
                    └─→ Knowledge Base ─┘
```

#### 3. Consensus Pattern (High Stakes)
```
                    ┌─→ RAG Verification ────┐
Claim ──────────────┼─→ Graph Reasoning ─────┤──→ Vote → Final Verdict
                    └─→ CoT Analysis ────────┘
```

#### 4. Dynamic Escalation
```
Fact-Checker → Low Confidence? ──Yes──→ Human Review
                     │
                     No
                     ↓
                  Reporter
```

---

## Orchestration Patterns

### Pattern 1: Conditional Routing

```python
def should_use_graph_reasoning(state: FactCheckingState) -> str:
    """Route to graph-based or standard reasoning"""
    complexity = state['classification'].get('complexity', 5)

    if complexity > 7:
        return "graph_fact_checker"
    else:
        return "standard_fact_checker"

workflow.add_conditional_edges(
    "classifier",
    should_use_graph_reasoning,
    {
        "graph_fact_checker": graph_fact_checker_node,
        "standard_fact_checker": standard_fact_checker_node
    }
)
```

### Pattern 2: Loop Until Confidence

```python
def check_confidence_threshold(state: FactCheckingState) -> str:
    """Continue retrieving evidence until confidence is high enough"""
    confidence = state.get('confidence', 0.0)
    iterations = state.get('iterations', 0)

    if confidence > 0.8 or iterations > 3:
        return "reporter"
    else:
        return "fact_checker"  # Loop back

workflow.add_conditional_edges(
    "fact_checker",
    check_confidence_threshold,
    {
        "fact_checker": fact_checker_node,
        "reporter": reporter_node
    }
)
```

---

## State Management

### State Lifecycle

```python
# 1. Initialize
state = FactCheckingState(
    original_claim="...",
    decomposed_assertions=[],
    # ... all fields with defaults
)

# 2. Agents modify state (immutable updates)
state = agent_classifier(state)  # Returns new state
state = agent_retriever(state)   # Returns new state

# 3. State persistence (checkpointing)
checkpoint = {
    "state": state,
    "timestamp": datetime.now(),
    "agent": "classifier"
}
redis.set(f"checkpoint:{claim_id}", checkpoint)

# 4. Recovery from failure
if failure_detected:
    checkpoint = redis.get(f"checkpoint:{claim_id}")
    state = checkpoint["state"]
    resume_from_agent = checkpoint["agent"]
```

### Memory Management

**Short-term Memory**: Within a single fact-checking session
```python
state['reasoning_trace'].append("Classifier: Found 3 assertions")
```

**Long-term Memory**: Across sessions
```python
# Store verified facts
fact_memory.add({
    "claim": "Paris is capital of France",
    "verdict": "SUPPORTED",
    "confidence": 0.99,
    "verified_at": "2025-01-15"
})

# Retrieve for future checks
similar_claims = fact_memory.search(
    query="What is the capital of France?",
    threshold=0.9
)
```

---

## Integration Points

### External APIs

#### Twitter/X API v2
```python
import tweepy

client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

def collect_tweets(query: str, max_results: int = 100):
    """Collect tweets matching query"""
    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=['created_at', 'author_id', 'public_metrics']
    )
    return tweets.data
```

#### Google Trends API (Alpha)
```python
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)

def get_trending_topics(timeframe='now 1-d'):
    """Get currently trending topics"""
    trending = pytrends.trending_searches(pn='united_states')
    return trending.head(10)
```

### Webhook Integrations

```python
# Notify external systems on verdict
@app.post("/webhooks/verdict")
async def send_verdict_webhook(claim_id: str, verdict: dict):
    """Send webhook notification"""

    for webhook_url in REGISTERED_WEBHOOKS:
        await httpx.post(
            webhook_url,
            json={
                "event": "fact_check_completed",
                "claim_id": claim_id,
                "verdict": verdict
            },
            headers={"X-Webhook-Secret": WEBHOOK_SECRET}
        )
```

---

## Scalability & Performance

### Horizontal Scaling

#### Agent-Level Scaling
```yaml
# docker-compose.yml
services:
  fact_checker:
    image: detect/fact-checker:latest
    deploy:
      replicas: 5  # Scale fact-checker to 5 instances
    environment:
      - WORKER_ID=${HOSTNAME}
```

#### Load Balancing
```
                    ┌─→ Fact-Checker Instance 1
Load Balancer ──────┼─→ Fact-Checker Instance 2
(NGINX)             └─→ Fact-Checker Instance 3
```

### Caching Strategy

```python
# Multi-level caching
class CacheStrategy:
    """
    L1: In-memory (LRU, 1000 items)
    L2: Redis (distributed, 1 hour TTL)
    L3: PostgreSQL (persistent)
    """

    async def get(self, key: str):
        # Check L1
        if key in self.l1_cache:
            return self.l1_cache[key]

        # Check L2
        value = await self.redis.get(key)
        if value:
            self.l1_cache[key] = value  # Promote to L1
            return value

        # Check L3
        value = await self.db.get(key)
        if value:
            await self.redis.set(key, value, ttl=3600)  # Promote to L2
            self.l1_cache[key] = value  # Promote to L1
            return value

        return None
```

### Performance Targets

| Operation | Target | Strategy |
|-----------|--------|----------|
| Evidence Retrieval | < 5s | Parallel fetching + caching |
| LLM Inference | < 10s | Batch processing + GPU |
| Database Write | < 100ms | Async writes + indexing |
| End-to-End | < 30s | Full pipeline optimization |

---

## Design Decisions

### Why LangGraph over CrewAI?

**Decision**: Use LangGraph as primary orchestrator

**Rationale**:
- **Fine-grained control**: Explicit state management and routing logic
- **Debugging**: Clear execution traces and checkpointing
- **Scalability**: Better support for distributed execution
- **Flexibility**: Easier to implement custom patterns (loops, conditionals)

**Trade-off**: Higher initial complexity vs CrewAI's simpler role-based approach

### Why Hybrid Retrieval (BM25 + Semantic)?

**Decision**: Combine sparse (BM25) and dense (embeddings) retrieval

**Rationale**:
- BM25 excellent for exact keyword matches (names, dates, numbers)
- Semantic search captures contextual meaning and synonyms
- Ensemble achieves **15% higher recall** than either alone (internal benchmarks)

**Trade-off**: 2x computational cost, but justified by accuracy gains

### Why PostgreSQL over MongoDB?

**Decision**: Use PostgreSQL as primary database

**Rationale**:
- ACID guarantees for critical fact-checking records
- Rich JSON support (JSONB) for flexible schemas
- Superior full-text search
- Better tooling ecosystem (pg_stat_statements, pgAdmin)

**Trade-off**: Slightly less flexible than MongoDB, but reliability is paramount

---

## Conclusion

This architecture balances:
- **Modularity**: Easy to extend with new agents/services
- **Performance**: < 30s end-to-end with caching and parallelization
- **Reliability**: Fault-tolerant with checkpointing and retries
- **Observability**: Comprehensive logging and metrics

For implementation details, see:
- [Agent Documentation](AGENTS.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT.md)
