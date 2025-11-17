# 🔍 Multi-Agent Misinformation Detection System

An intelligent, modular platform for automated detection and verification of misinformation using coordinated multi-agent workflows.

## 📋 Overview

This system employs **5 specialized AI agents** that collaborate to detect, analyze, and fact-check potential misinformation from social media and public sources in near real-time.

### Key Features

- ✅ **Automated Fact-Checking**: End-to-end claim verification pipeline
- 🤖 **Multi-Agent Orchestration**: 5 specialized agents working in coordination
- 🔬 **Semantic Anomaly Detection**: Identifies manipulative language patterns
- 📊 **Evidence Collection**: Multi-source aggregation (news, social media, fact-checking databases)
- 🎯 **Chain-of-Thought Reasoning**: LLM-powered verification with explainability
- 📈 **Real-time Dashboard**: Interactive Streamlit interface
- 🔌 **REST API**: Easy integration with external systems
- 🐳 **Containerized Deployment**: Docker + Docker Compose

---

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────┐
│         Fact-Checking Multi-Agent Workflow       │
└─────────────────────────────────────────────────┘

            ┌───────────────────┐
            │   1. CLASSIFIER   │
            │  - Categorize      │
            │  - Decompose       │
            │  - Extract NER     │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │   2. COLLECTOR    │
            │  - Web Search      │
            │  - News APIs       │
            │  - Social Media    │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ 3. ANOMALY DETECT │
            │  - Patterns        │
            │  - Coherence       │
            │  - Red Flags       │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │  4. FACT CHECKER  │
            │  - Verify          │
            │  - RAG             │
            │  - CoT Reasoning   │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │   5. REPORTER     │
            │  - Consolidate     │
            │  - Generate        │
            │  - Escalate?       │
            └───────────────────┘
```

### Agent Descriptions

| Agent | Role | Techniques |
|-------|------|-----------|
| **Classifier** | Categorize & decompose claims | NER, embeddings, thematic classification |
| **Collector** | Gather evidence from sources | API integration (Twitter, Reddit, News), web scraping |
| **Anomaly Detector** | Identify suspicious patterns | Manipulation detection, emotional analysis, coherence checking |
| **Fact Checker** | Verify claims with evidence | RAG (Retrieval-Augmented Generation), Chain-of-Thought, credibility scoring |
| **Reporter** | Generate reports & alerts | Consolidation, confidence scoring, escalation logic |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for containerized deployment)
- API keys (optional):
  - OpenAI / Anthropic / Mistral (for LLM features)
  - Twitter/X API
  - Reddit API
  - News API

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/detect.git
cd detect
```

2. **Set up environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

3. **Install dependencies**

**Option A: Using Python virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Using Docker Compose (recommended)**
```bash
docker-compose up --build
```

### Running the System

#### Local Development

**1. Start the API**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the Dashboard**
```bash
streamlit run dashboard/app.py
```

**3. Run Example Script**
```bash
python scripts/run_example.py
```

**4. Access the services**
- API Documentation: http://localhost:8000/docs
- Dashboard: http://localhost:8501

#### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📚 Usage

### API Examples

**Submit a claim for verification**

```bash
curl -X POST "http://localhost:8000/api/v1/fact-check" \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "The president announced new climate policies yesterday",
    "priority": "normal"
  }'
```

**Response:**
```json
{
  "claim_id": "a1b2c3d4e5f6g7h8",
  "verdict": "INSUFFICIENT_INFO",
  "confidence": 0.65,
  "created_at": "2025-01-15T10:30:00",
  "evidence_summary": [
    {
      "domain": "bbc.com",
      "credibility": 0.95,
      "url": "https://bbc.com/news/..."
    }
  ]
}
```

**Get verification result**

```bash
curl "http://localhost:8000/api/v1/fact-check/a1b2c3d4e5f6g7h8"
```

### Python SDK Example

```python
from agents.orchestrator import create_orchestrator

# Initialize orchestrator
orchestrator = create_orchestrator()

# Check a claim
result = await orchestrator.check_claim(
    claim="The president announced new climate policies yesterday",
    priority="normal"
)

print(f"Verdict: {result.final_verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Evidence sources: {len(result.evidence_retrieved)}")
```

---

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Configuration
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Social Media APIs
TWITTER_BEARER_TOKEN=your_token
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret

# Database
POSTGRES_HOST=localhost
POSTGRES_DB=factcheck_db
POSTGRES_USER=factcheck_user
POSTGRES_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Application
LOG_LEVEL=INFO
FACT_CHECK_TIMEOUT=30
```

---

## 📊 Project Structure

```
detect/
├── agents/                 # Multi-agent system
│   ├── base_agent.py       # Base agent class
│   ├── classifier_agent.py # Agent 1: Classification
│   ├── collector_agent.py  # Agent 2: Collection
│   ├── anomaly_detector_agent.py # Agent 3: Anomaly detection
│   ├── fact_checker_agent.py     # Agent 4: Fact-checking
│   ├── reporter_agent.py   # Agent 5: Reporting
│   └── orchestrator.py     # LangGraph orchestration
├── api/                    # FastAPI application
│   └── main.py
├── dashboard/              # Streamlit dashboard
│   └── app.py
├── config/                 # Configuration
│   └── settings.py
├── utils/                  # Utilities
│   ├── types.py            # Type definitions
│   ├── logger.py           # Logging setup
│   ├── helpers.py          # Helper functions
│   └── credibility.py      # Credibility scoring
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
│   └── run_example.py      # Example usage
├── data/                   # Data storage
├── logs/                   # Application logs
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

---

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov=api --cov=utils

# Run specific test file
pytest tests/test_classifier_agent.py -v
```

---

## 📈 Performance Metrics

Target metrics for production deployment:

| Metric | Target |
|--------|--------|
| **Classification Accuracy** | > 90% |
| **F1-Score (Fact-checking)** | > 0.85 |
| **Average Latency** | < 30s |
| **False Positive Rate** | < 5% |
| **Throughput** | 1000 claims/hour |

---

## 🛣️ Roadmap

### Phase 1: MVP ✅
- [x] Basic multi-agent architecture
- [x] Core 5 agents implemented
- [x] FastAPI REST API
- [x] Streamlit dashboard
- [x] Docker deployment

### Phase 2: Enhanced Features
- [ ] LLM integration (Claude, GPT-4)
- [ ] Vector database (RAG) for evidence retrieval
- [ ] Social media API integrations
- [ ] Deepfake detection (image/video)
- [ ] Database persistence (PostgreSQL)

### Phase 3: Production
- [ ] Kubernetes deployment
- [ ] Advanced monitoring (Prometheus, Grafana)
- [ ] User authentication & authorization
- [ ] Rate limiting & quotas
- [ ] Advanced caching strategies
- [ ] Multi-language support

---

## 📄 License

This project is licensed under the MIT License.

---

## 📚 References

### Academic Papers
- Multi-Agent Debate for Misinformation Detection (2025)
- FACT-AUDIT: Adaptive Multi-Agent Framework (ACL 2025)
- Toward Verifiable Misinformation Detection (ArXiv 2025)

### Technical Documentation
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Note**: This is a research prototype. Always verify critical information with trusted, authoritative sources.