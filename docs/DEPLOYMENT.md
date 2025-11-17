# Deployment Guide

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Database Migration](#database-migration)
6. [Scaling Strategy](#scaling-strategy)
7. [Rollback Procedures](#rollback-procedures)
8. [Production Checklist](#production-checklist)

---

## Deployment Overview

### Deployment Environments

| Environment | Purpose | Update Frequency |
|-------------|---------|------------------|
| **Development** | Feature development | Continuous |
| **Staging** | Pre-production testing | Daily |
| **Production** | Live system | Weekly/On-demand |

### Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Load Balancer (NGINX)               │
│               (SSL Termination, Rate Limiting)       │
└──────────────────┬──────────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────▼──────┐       ┌───────▼──────┐
│   API Pod 1  │       │   API Pod 2  │  (Auto-scaling: 2-10 pods)
└──────┬──────┘       └───────┬──────┘
       │                       │
       └───────────┬───────────┘
                   │
       ┌───────────┴──────────┬─────────────┐
       │                      │             │
┌──────▼──────┐   ┌──────────▼───┐   ┌─────▼──────┐
│ PostgreSQL  │   │    Redis      │   │  Weaviate  │
│ (Primary+   │   │  (Cluster)    │   │  (Vector)  │
│  Replica)   │   │               │   │            │
└─────────────┘   └──────────────┘   └────────────┘
```

---

## Environment Setup

### Production Environment Variables

```bash
# .env.production

# Application
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
WORKERS=4

# Database (Read-Write)
DATABASE_URL=postgresql://user:pass@postgres-primary.internal:5432/detect_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
DATABASE_SSL_MODE=require

# Database (Read-Only Replica)
DATABASE_REPLICA_URL=postgresql://user:pass@postgres-replica.internal:5432/detect_prod

# Redis Cluster
REDIS_CLUSTER_NODES=redis-1.internal:6379,redis-2.internal:6379,redis-3.internal:6379
REDIS_PASSWORD=<secure-password>
REDIS_SSL=true

# Vector Database
WEAVIATE_URL=https://weaviate.internal:8080
WEAVIATE_API_KEY=<secure-key>

# LLM APIs (use secrets manager)
ANTHROPIC_API_KEY=<from-secrets-manager>
OPENAI_API_KEY=<from-secrets-manager>

# External APIs
TWITTER_BEARER_TOKEN=<from-secrets-manager>
REDDIT_CLIENT_ID=<from-secrets-manager>
REDDIT_CLIENT_SECRET=<from-secrets-manager>

# Security
SECRET_KEY=<256-bit-random-key>
ENCRYPTION_KEY=<256-bit-random-key>
ALLOWED_HOSTS=api.detect-project.com
CORS_ORIGINS=https://detect-project.com,https://dashboard.detect-project.com

# Monitoring
SENTRY_DSN=https://xxx@sentry.io/xxx
PROMETHEUS_ENABLED=true
METRICS_PORT=9090

# Feature Flags
ENABLE_DEEPFAKE_DETECTION=true
ENABLE_GRAPH_REASONING=true
ENABLE_RATE_LIMITING=true

# Performance
CACHE_TTL=3600
MAX_REQUEST_SIZE_MB=10
REQUEST_TIMEOUT_SECONDS=120
```

### Secrets Management

```bash
# Using AWS Secrets Manager
aws secretsmanager create-secret \
    --name detect/production/anthropic-key \
    --secret-string "sk-ant-xxxxx"

# Retrieve in application
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-west-2')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

ANTHROPIC_API_KEY = get_secret('detect/production/anthropic-key')
```

---

## Docker Deployment

### Production Dockerfile

```dockerfile
# Multi-stage build for optimized image size

# Stage 1: Build
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY detect/ ./detect/
COPY config/ ./config/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create non-root user
RUN useradd -m -u 1000 detect && chown -R detect:detect /app
USER detect

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "detect.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Production

```yaml
# docker-compose.prod.yml

version: '3.8'

services:
  api:
    image: detect/api:${VERSION}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        max_attempts: 3
    environment:
      - APP_ENV=production
    env_file:
      - .env.production
    networks:
      - detect-network
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    networks:
      - detect-network
    depends_on:
      - api

networks:
  detect-network:
    driver: overlay
```

---

## Kubernetes Deployment

### Kubernetes Manifests

#### Deployment

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: detect-api
  namespace: detect
  labels:
    app: detect-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: detect-api
  template:
    metadata:
      labels:
        app: detect-api
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: detect/api:1.0.0
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: APP_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: detect-secrets
              key: database-url
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: detect-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - detect-api
              topologyKey: kubernetes.io/hostname
```

#### Service

```yaml
# k8s/service.yaml

apiVersion: v1
kind: Service
metadata:
  name: detect-api
  namespace: detect
  labels:
    app: detect-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: detect-api
```

#### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: detect-api-hpa
  namespace: detect
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: detect-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace detect

# Create secrets
kubectl create secret generic detect-secrets \
    --from-literal=database-url="postgresql://..." \
    --from-literal=anthropic-api-key="sk-ant-..." \
    --namespace=detect

# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n detect
kubectl get svc -n detect
kubectl get hpa -n detect

# Check logs
kubectl logs -f deployment/detect-api -n detect
```

---

## Database Migration

### Migration Strategy

```bash
# Run migrations before deploying new version
alembic upgrade head

# Verify migration
alembic current
alembic history

# Rollback if needed
alembic downgrade -1
```

### Zero-Downtime Migration

```python
# Migration best practices

# 1. Additive changes (safe)
def upgrade():
    # Add new column (nullable first)
    op.add_column('fact_checks', sa.Column('new_field', sa.String(), nullable=True))

def downgrade():
    op.drop_column('fact_checks', 'new_field')

# 2. Backfill data
def upgrade():
    # Add column
    op.add_column('fact_checks', sa.Column('cached_score', sa.Float(), nullable=True))

    # Backfill existing rows
    connection = op.get_bind()
    connection.execute(
        text("UPDATE fact_checks SET cached_score = confidence WHERE cached_score IS NULL")
    )

    # Make non-nullable
    op.alter_column('fact_checks', 'cached_score', nullable=False)

# 3. Multi-step for breaking changes
# Step 1: Add new column
# Step 2: Dual-write to both columns
# Step 3: Backfill old data
# Step 4: Switch reads to new column
# Step 5: Remove old column
```

---

## Scaling Strategy

### Horizontal Scaling

```yaml
# Scale API pods
kubectl scale deployment detect-api --replicas=5 -n detect

# Auto-scaling based on custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: detect-api-custom-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: detect-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### Database Scaling

```yaml
# Read Replicas
DATABASE_PRIMARY_URL=postgresql://primary.internal:5432/detect_prod
DATABASE_REPLICA_URLS=
  - postgresql://replica-1.internal:5432/detect_prod
  - postgresql://replica-2.internal:5432/detect_prod

# Connection Pooling
class DatabaseRouter:
    """Route reads to replicas, writes to primary"""

    def db_for_read(self):
        return random.choice(REPLICA_URLS)

    def db_for_write(self):
        return PRIMARY_URL
```

### Caching Strategy

```python
# Multi-tier caching
class CachingStrategy:
    """
    L1: In-memory (fast, limited size)
    L2: Redis (distributed, larger)
    L3: Database (persistent)
    """

    def get_with_caching(self, key: str):
        # Check L1
        if key in memory_cache:
            return memory_cache[key]

        # Check L2
        value = redis.get(key)
        if value:
            memory_cache[key] = value
            return value

        # Check L3
        value = database.get(key)
        if value:
            redis.setex(key, 3600, value)
            memory_cache[key] = value
            return value

        return None
```

---

## Rollback Procedures

### Quick Rollback

```bash
# Kubernetes rollback
kubectl rollout undo deployment/detect-api -n detect

# Rollback to specific revision
kubectl rollout history deployment/detect-api -n detect
kubectl rollout undo deployment/detect-api --to-revision=2 -n detect

# Docker Swarm rollback
docker service rollback detect-api
```

### Database Rollback

```bash
# Alembic rollback
alembic downgrade -1  # One version back
alembic downgrade <revision>  # Specific version

# Backup restoration
pg_restore -d detect_prod -c backup_20250115.dump
```

### Blue-Green Deployment

```yaml
# Maintain two identical environments
# Switch traffic between them

# Current (Blue)
apiVersion: v1
kind: Service
metadata:
  name: detect-api
spec:
  selector:
    app: detect-api
    version: blue

# New (Green)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: detect-api-green
spec:
  selector:
    matchLabels:
      version: green

# Switch traffic
kubectl patch service detect-api -p '{"spec":{"selector":{"version":"green"}}}'
```

---

## Production Checklist

### Pre-Deployment

- [ ] All tests passing (unit, integration, e2e)
- [ ] Code reviewed and approved
- [ ] Security scan completed (no critical vulnerabilities)
- [ ] Performance testing completed
- [ ] Database migrations tested in staging
- [ ] Environment variables configured
- [ ] Secrets rotated and stored securely
- [ ] Monitoring and alerting configured
- [ ] Backup strategy verified
- [ ] Rollback plan documented
- [ ] Stakeholders notified

### Deployment

- [ ] Database backup created
- [ ] Run database migrations
- [ ] Deploy new version (rolling update)
- [ ] Verify health checks passing
- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Verify external integrations
- [ ] Test critical user flows

### Post-Deployment

- [ ] Monitor logs for errors
- [ ] Check performance metrics (latency, throughput)
- [ ] Verify auto-scaling works
- [ ] Test rollback procedure
- [ ] Update documentation
- [ ] Notify stakeholders of completion
- [ ] Schedule post-mortem (if issues)

---

## CI/CD Pipeline

```yaml
# .github/workflows/deploy-production.yml

name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Tests
        run: |
          pytest tests/ --cov=detect

      - name: Security Scan
        run: |
          pip-audit
          safety check

      - name: Build Docker Image
        run: |
          docker build -t detect/api:${{ github.ref_name }} .

      - name: Push to Registry
        run: |
          docker push detect/api:${{ github.ref_name }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/detect-api \
            api=detect/api:${{ github.ref_name }} \
            -n detect

      - name: Wait for Rollout
        run: |
          kubectl rollout status deployment/detect-api -n detect --timeout=5m

      - name: Smoke Tests
        run: |
          ./scripts/smoke-tests.sh

      - name: Notify Team
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Production deployment completed'
```

---

**Deployment Contact**: devops@detect-project.com
**Last Deployment**: 2025-01-15
**Next Scheduled**: Weekly releases every Monday
