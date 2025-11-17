# Installation Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Configuration](#configuration)
5. [Docker Setup](#docker-setup)
6. [Database Setup](#database-setup)
7. [Service Dependencies](#service-dependencies)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ (with WSL2) |
| **Python** | 3.10 or higher |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **Storage** | 20 GB free space |
| **CPU** | 4 cores minimum, 8 cores recommended |
| **GPU** | Optional (NVIDIA with CUDA 11.8+ for faster inference) |

### Software Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- PostgreSQL 14+
- Redis 7+
- Git 2.30+
- Node.js 18+ (for frontend dashboard)

---

## Quick Start

### One-Command Setup (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/yourusername/detect.git
cd detect

# Run automated setup script
./scripts/setup.sh

# Start all services
docker-compose up -d

# Verify installation
python scripts/verify_installation.py
```

The setup script will:
- ✅ Check system requirements
- ✅ Create virtual environment
- ✅ Install Python dependencies
- ✅ Setup PostgreSQL database
- ✅ Configure Redis
- ✅ Download required models
- ✅ Initialize environment variables

---

## Detailed Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/detect.git
cd detect
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n detect python=3.10
conda activate detect
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Verify installation
pip list | grep detect
```

**requirements.txt** (main dependencies):
```txt
# Core Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# Orchestration
langgraph==0.0.40
langchain==0.1.4
langchain-community==0.0.16
crewai==0.1.2

# LLMs
anthropic==0.8.1
openai==1.10.0
tiktoken==0.5.2

# NLP
spacy==3.7.2
transformers==4.37.0
sentence-transformers==2.3.1
torch==2.1.2

# Vector DB
weaviate-client==3.26.0
pinecone-client==3.0.0

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.25
alembic==1.13.1
redis==5.0.1

# Web & APIs
httpx==0.26.0
requests==2.31.0
beautifulsoup4==4.12.3
tweepy==4.14.0
praw==7.7.1

# Data Processing
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0

# Monitoring
prometheus-client==0.19.0
sentry-sdk==1.40.0

# Utils
python-dotenv==1.0.0
pyyaml==6.0.1
click==8.1.7
```

### Step 4: Download NLP Models

```bash
# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# Download sentence transformer models (cached on first use)
python scripts/download_models.py
```

### Step 5: Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your favorite editor
```

**.env** template:
```env
# Application
APP_ENV=development
APP_PORT=8000
APP_HOST=0.0.0.0
DEBUG=true

# Database
DATABASE_URL=postgresql://detect_user:password@localhost:5432/detect_db
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# Vector Database (Weaviate)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_weaviate_key

# Or Pinecone (alternative)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-west1-gcp

# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
MISTRAL_API_KEY=xxxxx

# External APIs
TWITTER_BEARER_TOKEN=xxxxx
REDDIT_CLIENT_ID=xxxxx
REDDIT_CLIENT_SECRET=xxxxx
NEWSAPI_KEY=xxxxx
GOOGLE_TRENDS_API_KEY=xxxxx

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
API_KEY_PREFIX=sk-proj-

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_DAY=50000

# Monitoring
SENTRY_DSN=https://xxxxx@sentry.io/xxxxx
PROMETHEUS_PORT=9090

# Feature Flags
ENABLE_DEEPFAKE_DETECTION=true
ENABLE_GRAPH_REASONING=true
ENABLE_MULTIMODAL_ANALYSIS=true
```

---

## Docker Setup

### Using Docker Compose (Recommended)

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    container_name: detect-postgres
    environment:
      POSTGRES_DB: detect_db
      POSTGRES_USER: detect_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U detect_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: detect-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Weaviate Vector Database
  weaviate:
    image: semitechnologies/weaviate:1.23.0
    container_name: detect-weaviate
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
    volumes:
      - weaviate_data:/var/lib/weaviate

  # Transformers Inference
  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    container_name: detect-transformers
    environment:
      ENABLE_CUDA: '0'

  # RabbitMQ Message Queue
  rabbitmq:
    image: rabbitmq:3.12-management-alpine
    container_name: detect-rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: detect
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  # FastAPI Backend
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: detect-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://detect_user:${DB_PASSWORD}@postgres:5432/detect_db
      - REDIS_URL=redis://redis:6379/0
      - WEAVIATE_URL=http://weaviate:8080
    volumes:
      - ./detect:/app/detect
      - ./config:/app/config
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: uvicorn detect.main:app --host 0.0.0.0 --port 8000 --reload

  # Streamlit Dashboard
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: detect-dashboard
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    volumes:
      - ./dashboard:/app/dashboard
    depends_on:
      - api

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: detect-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: detect-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning

volumes:
  postgres_data:
  redis_data:
  weaviate_data:
  rabbitmq_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: detect-network
```

### Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Check service status
docker-compose ps

# Stop services
docker-compose down

# Reset everything (WARNING: deletes data)
docker-compose down -v
```

---

## Database Setup

### Initialize PostgreSQL

```bash
# Create database and user
psql -U postgres << EOF
CREATE DATABASE detect_db;
CREATE USER detect_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE detect_db TO detect_user;
EOF

# Run migrations
cd detect
alembic upgrade head
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# View migration history
alembic history
```

### Seed Initial Data

```bash
# Load source credibility scores
python scripts/seed_sources.py

# Load fact-check database references
python scripts/seed_factcheck_dbs.py

# Create test data (development only)
python scripts/create_test_data.py
```

---

## Service Dependencies

### Install System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    git \
    curl \
    wget
```

### Install System Dependencies (macOS)

```bash
brew update
brew install \
    postgresql@14 \
    redis \
    python@3.10 \
    git
```

### GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# Install CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install cuDNN
# Download from https://developer.nvidia.com/cudnn
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Verification

### Run Verification Script

```bash
python scripts/verify_installation.py
```

**Expected Output**:
```
✅ Python version: 3.10.12
✅ PostgreSQL connection: OK
✅ Redis connection: OK
✅ Weaviate connection: OK
✅ Required packages installed: 45/45
✅ Environment variables set: 28/28
✅ NLP models downloaded: en_core_web_sm, en_core_web_lg
✅ GPU available: Yes (NVIDIA GeForce RTX 3090)
✅ API server: http://localhost:8000 (healthy)
✅ Dashboard: http://localhost:8501 (healthy)

All checks passed! Installation successful.
```

### Manual Verification

#### Test Database Connection
```bash
python << EOF
from sqlalchemy import create_engine
engine = create_engine("postgresql://detect_user:password@localhost:5432/detect_db")
conn = engine.connect()
print("Database connection: OK")
conn.close()
EOF
```

#### Test Redis Connection
```bash
python << EOF
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
r.set('test', 'ok')
print(f"Redis connection: {r.get('test').decode()}")
EOF
```

#### Test API
```bash
# Health check
curl http://localhost:8000/v1/health

# Test fact-check endpoint
curl -X POST http://localhost:8000/v1/fact-check \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_key" \
  -d '{"claim": "Test claim"}'
```

---

## Troubleshooting

### Common Issues

#### Issue: Port Already in Use

```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Or change port in .env
APP_PORT=8001
```

#### Issue: PostgreSQL Connection Failed

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Check connection
psql -U detect_user -d detect_db -h localhost
```

#### Issue: Redis Connection Failed

```bash
# Check if Redis is running
sudo systemctl status redis

# Start Redis
sudo systemctl start redis

# Test connection
redis-cli ping
```

#### Issue: Out of Memory

```bash
# Increase Docker memory limit
# Edit Docker Desktop settings or:
docker system prune -a  # Clean up unused resources

# Reduce worker processes in .env
WORKER_PROCESSES=2
```

#### Issue: Model Download Fails

```bash
# Download models manually
mkdir -p models
cd models

# Download sentence transformer
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin

# Set model path in .env
MODEL_CACHE_DIR=./models
```

#### Issue: Permission Denied

```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod -R 755 scripts/

# Make scripts executable
chmod +x scripts/*.sh
```

### Logs and Debugging

```bash
# View API logs
docker-compose logs -f api

# View all service logs
docker-compose logs -f

# Check disk space
df -h

# Check memory usage
free -h

# View process resource usage
htop
```

### Getting Help

If you encounter issues not covered here:

1. **Check documentation**: [https://docs.detect-project.com](https://docs.detect-project.com)
2. **Search existing issues**: [GitHub Issues](https://github.com/yourusername/detect/issues)
3. **Join Discord**: [https://discord.gg/detect](https://discord.gg/detect)
4. **Create new issue**: Include logs, system info, and steps to reproduce

---

## Next Steps

After successful installation:

1. **Configure API Keys**: Add your LLM and external API keys to `.env`
2. **Read Development Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
3. **Explore Examples**: Check `examples/` directory
4. **Run Tests**: `pytest tests/`
5. **Try Demo**: `python examples/demo_fact_check.py`

---

**Installation Guide Version**: 1.0
**Last Updated**: 2025-01-15
