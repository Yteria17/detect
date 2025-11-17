# Quick Start Guide - Phase 3

Get the Multi-Agent Fact-Checking System up and running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- OpenAI or Anthropic API key (for LLM access)

## Step 1: Clone and Setup

```bash
cd detect

# The quickstart script will create .env from .env.example
./quickstart.sh start
```

## Step 2: Configure API Keys

Edit the `.env` file and add your API keys:

```bash
nano .env

# Add:
OPENAI_API_KEY=sk-your-key-here
# or
ANTHROPIC_API_KEY=your-key-here
```

## Step 3: Restart Services

```bash
./quickstart.sh restart
```

## Step 4: Test the API

```bash
# Check health
curl http://localhost:8000/health

# Submit a fact-check
curl -X POST http://localhost:8000/api/v1/fact-check \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "The Earth is flat",
    "priority": "normal"
  }'

# You'll get a response like:
# {
#   "claim_id": "claim_abc123",
#   "status": "pending",
#   ...
# }

# Check the result (wait a few seconds first)
curl http://localhost:8000/api/v1/fact-check/claim_abc123
```

## Step 5: Explore the System

### Access the Interactive API Documentation

Open your browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### View Monitoring Dashboards

- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`

### View Metrics

- **Prometheus Metrics**: http://localhost:9090/metrics

## Common Commands

```bash
# Start services
./quickstart.sh start

# Stop services
./quickstart.sh stop

# View logs
./quickstart.sh logs

# Check health
./quickstart.sh health

# Run tests
./quickstart.sh test
```

## Example API Usage

### Python Example

```python
import requests
import time

# Submit fact-check
response = requests.post(
    "http://localhost:8000/api/v1/fact-check",
    json={
        "claim": "Water boils at 100°C at sea level",
        "priority": "normal"
    }
)

claim_id = response.json()["claim_id"]
print(f"Submitted claim: {claim_id}")

# Wait for processing
time.sleep(3)

# Get result
result = requests.get(
    f"http://localhost:8000/api/v1/fact-check/{claim_id}"
).json()

print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Evidence: {result['evidence_count']} sources")
print(f"Key Findings: {result['key_findings']}")
```

### JavaScript Example

```javascript
// Submit fact-check
const response = await fetch('http://localhost:8000/api/v1/fact-check', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    claim: "The sun is a star",
    priority: "normal"
  })
});

const { claim_id } = await response.json();
console.log('Claim ID:', claim_id);

// Wait and get result
setTimeout(async () => {
  const result = await fetch(
    `http://localhost:8000/api/v1/fact-check/${claim_id}`
  ).then(r => r.json());

  console.log('Verdict:', result.verdict);
  console.log('Confidence:', result.confidence);
}, 3000);
```

## Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker --version
docker-compose --version

# View logs for errors
docker-compose logs
```

### API returns errors

```bash
# Check .env file has API keys
cat .env | grep API_KEY

# Restart services
./quickstart.sh restart
```

### Low performance

```bash
# Increase API workers in docker-compose.yml
# Edit the command line:
command: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Restart
./quickstart.sh restart
```

## Next Steps

1. Read the full documentation in `README_PHASE3.md`
2. Explore the API endpoints in Swagger UI
3. Set up custom Grafana dashboards
4. Run the benchmark suite: `python tests/benchmarks.py`
5. Integrate with your application

## Support

For detailed documentation, see:
- `README_PHASE3.md` - Complete documentation
- `projet-multi-agents-desinformation.md` - Project overview
- `technique-approfondi.md` - Technical deep-dive

Happy fact-checking! 🎯
