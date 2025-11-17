# Monitoring & Observability Guide

## Table of Contents

1. [Observability Overview](#observability-overview)
2. [Metrics Collection](#metrics-collection)
3. [Logging Strategy](#logging-strategy)
4. [Distributed Tracing](#distributed-tracing)
5. [Alerting](#alerting)
6. [Dashboards](#dashboards)
7. [Performance Monitoring](#performance-monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Observability Overview

### Three Pillars of Observability

```
┌────────────────────────────────────────────────────┐
│              OBSERVABILITY STACK                    │
├────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ METRICS  │  │   LOGS   │  │  TRACES  │        │
│  │          │  │          │  │          │        │
│  │Prometheus│  │   ELK    │  │  Jaeger  │        │
│  │  Grafana │  │ Loki     │  │  Tempo   │        │
│  └──────────┘  └──────────┘  └──────────┘        │
│                                                     │
│  ┌────────────────────────────────────────┐       │
│  │      Visualization: Grafana             │       │
│  └────────────────────────────────────────┘       │
│                                                     │
│  ┌────────────────────────────────────────┐       │
│  │      Alerting: PagerDuty / Slack        │       │
│  └────────────────────────────────────────┘       │
└────────────────────────────────────────────────────┘
```

---

## Metrics Collection

### Prometheus Instrumentation

```python
# detect/utils/metrics.py

from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Request metrics
request_count = Counter(
    'fact_check_requests_total',
    'Total number of fact-check requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'fact_check_request_duration_seconds',
    'Fact-check request duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Agent metrics
agent_invocations = Counter(
    'agent_invocations_total',
    'Number of times each agent was invoked',
    ['agent_name', 'status']
)

agent_duration = Histogram(
    'agent_duration_seconds',
    'Agent execution duration',
    ['agent_name'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
)

# Verdict metrics
verdicts_by_type = Counter(
    'verdicts_total',
    'Number of verdicts by type',
    ['verdict']  # SUPPORTED/REFUTED/INSUFFICIENT_INFO
)

confidence_scores = Histogram(
    'confidence_scores',
    'Distribution of confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# System metrics
active_requests = Gauge(
    'active_requests',
    'Number of requests currently being processed'
)

llm_api_calls = Counter(
    'llm_api_calls_total',
    'Total LLM API calls',
    ['provider', 'model', 'status']
)

# Application info
app_info = Info('detect_app', 'Application information')
app_info.info({
    'version': '1.0.0',
    'environment': 'production'
})
```

### Metrics Middleware

```python
# detect/api/middleware/metrics.py

from starlette.middleware.base import BaseHTTPMiddleware
import time

class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect HTTP request metrics"""

    async def dispatch(self, request, call_next):
        # Increment active requests
        active_requests.inc()

        # Record start time
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Record metrics
            duration = time.time() - start_time
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)

            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()

            return response

        finally:
            # Decrement active requests
            active_requests.dec()

app.add_middleware(MetricsMiddleware)
```

### Metrics Endpoint

```python
# detect/api/v1/monitoring.py

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### Prometheus Configuration

```yaml
# config/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'detect-api'
    static_configs:
      - targets: ['api:9090']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - '/etc/prometheus/rules/*.yml'
```

---

## Logging Strategy

### Structured Logging

```python
# detect/utils/logger.py

import logging
import json
from datetime import datetime

class StructuredLogger:
    """JSON structured logging"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def info(self, message: str, **kwargs):
        """Log info with structured context"""
        self.logger.info(message, extra=kwargs)

    def error(self, message: str, exc_info=None, **kwargs):
        """Log error with exception info"""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)

class JsonFormatter(logging.Formatter):
    """Format logs as JSON"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields
        if hasattr(record, 'claim_id'):
            log_data['claim_id'] = record.claim_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Usage
logger = StructuredLogger(__name__)

logger.info(
    "Fact-check completed",
    claim_id="abc123",
    verdict="REFUTED",
    confidence=0.87,
    duration_ms=24500
)
```

### Log Levels

| Level | Use Case |
|-------|----------|
| **DEBUG** | Detailed diagnostic info (development only) |
| **INFO** | General informational messages |
| **WARNING** | Warning messages (degraded performance) |
| **ERROR** | Error messages (operation failed) |
| **CRITICAL** | Critical errors (system down) |

### ELK Stack Configuration

```yaml
# docker-compose.monitoring.yml

version: '3.8'

services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  logstash:
    image: logstash:8.11.0
    ports:
      - "5044:5044"
    volumes:
      - ./config/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - elasticsearch
```

---

## Distributed Tracing

### OpenTelemetry Integration

```python
# detect/utils/tracing.py

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Manual instrumentation
def process_claim_with_tracing(claim: str) -> Result:
    """Process claim with distributed tracing"""

    with tracer.start_as_current_span("process_claim") as span:
        span.set_attribute("claim.length", len(claim))
        span.set_attribute("claim.id", claim_id)

        # Collector
        with tracer.start_as_current_span("agent.collector"):
            collector_result = collector_agent.process(state)

        # Classifier
        with tracer.start_as_current_span("agent.classifier"):
            classifier_result = classifier_agent.process(state)

        # Fact-checker
        with tracer.start_as_current_span("agent.fact_checker"):
            # Nested span for LLM call
            with tracer.start_as_current_span("llm.generate"):
                llm_response = llm.generate(prompt)

        span.set_attribute("verdict", result.verdict)
        span.set_attribute("confidence", result.confidence)

        return result
```

---

## Alerting

### Alert Rules

```yaml
# config/prometheus/rules/alerts.yml

groups:
  - name: detect_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(fact_check_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(fact_check_request_duration_seconds_bucket[5m])) > 30
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s"

      # Low confidence rate
      - alert: LowConfidenceRate
        expr: |
          rate(confidence_scores_bucket{le="0.5"}[1h]) / rate(confidence_scores_count[1h]) > 0.3
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "High rate of low-confidence verdicts"
          description: "{{ $value | humanizePercentage }} of verdicts have confidence < 0.5"

      # Service down
      - alert: ServiceDown
        expr: up{job="detect-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} is unreachable"

      # Database connection issues
      - alert: DatabaseConnectionPoolExhausted
        expr: |
          database_pool_size - database_pool_available < 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool nearly exhausted"
```

### Alert Notifications

```python
# detect/utils/alerting.py

import requests

class AlertService:
    """Send alerts to various channels"""

    def __init__(self):
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.pagerduty_key = os.getenv('PAGERDUTY_API_KEY')

    def send_alert(self, alert: Alert):
        """Send alert to all configured channels"""
        if alert.severity == 'critical':
            self.send_to_pagerduty(alert)
            self.send_to_slack(alert, urgent=True)
        elif alert.severity == 'warning':
            self.send_to_slack(alert, urgent=False)
        else:
            self.log_alert(alert)

    def send_to_slack(self, alert: Alert, urgent: bool = False):
        """Send alert to Slack"""
        color = '#ff0000' if urgent else '#ffaa00'

        payload = {
            "attachments": [{
                "color": color,
                "title": f"🚨 {alert.title}" if urgent else f"⚠️ {alert.title}",
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity, "short": True},
                    {"title": "Service", "value": "detect-api", "short": True},
                ],
                "footer": "Detect Monitoring",
                "ts": int(time.time())
            }]
        }

        requests.post(self.slack_webhook, json=payload)

    def send_to_pagerduty(self, alert: Alert):
        """Trigger PagerDuty incident"""
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "severity": alert.severity,
                "source": "detect-api",
                "custom_details": {
                    "description": alert.description
                }
            }
        }

        requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload
        )
```

---

## Dashboards

### Grafana Dashboard (JSON)

```json
{
  "dashboard": {
    "title": "Detect - Fact Checking System",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fact_check_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Latency (p50, p95, p99)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(fact_check_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(fact_check_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(fact_check_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Verdict Distribution",
        "type": "pie",
        "targets": [
          {
            "expr": "sum by (verdict) (verdicts_total)",
            "legendFormat": "{{verdict}}"
          }
        ]
      },
      {
        "title": "Agent Performance",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(agent_duration_seconds_bucket[5m])",
            "legendFormat": "{{agent_name}}"
          }
        ]
      }
    ]
  }
}
```

---

## Performance Monitoring

### APM (Application Performance Monitoring)

```python
# Sentry integration for error tracking

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    environment=os.getenv('APP_ENV'),
    traces_sample_rate=0.1,  # Sample 10% of transactions
    integrations=[FastApiIntegration()],
    before_send=filter_sensitive_data
)

def filter_sensitive_data(event, hint):
    """Remove sensitive data before sending to Sentry"""
    if 'request' in event:
        # Remove API keys from headers
        if 'headers' in event['request']:
            event['request']['headers'].pop('Authorization', None)

    return event
```

### Custom Performance Tracking

```python
class PerformanceTracker:
    """Track performance metrics"""

    def __init__(self):
        self.metrics = defaultdict(list)

    @contextmanager
    def track(self, operation: str):
        """Context manager to track operation duration"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.metrics[operation].append(duration)
            self.report_metric(operation, duration)

    def report_metric(self, operation: str, duration: float):
        """Report metric to monitoring system"""
        # Prometheus
        request_duration.labels(operation=operation).observe(duration)

        # Slow operation warning
        if duration > SLOW_THRESHOLD:
            logger.warning(
                f"Slow operation detected: {operation}",
                duration_ms=duration * 1000
            )

# Usage
tracker = PerformanceTracker()

with tracker.track("evidence_retrieval"):
    evidence = retriever.get_relevant_documents(query)
```

---

## Troubleshooting

### Common Issues

#### High Latency

1. Check metrics dashboard
2. Identify slow component (agent, LLM, database)
3. Review traces for bottlenecks
4. Check database query performance
5. Review cache hit rates

```bash
# Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

#### Memory Leaks

```bash
# Monitor memory usage
kubectl top pods -n detect

# Get detailed metrics
kubectl exec -it pod-name -- python -m memory_profiler detect/main.py
```

#### Database Connection Pool Exhausted

```python
# Monitor pool status
from sqlalchemy import event

@event.listens_for(Engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    pool = dbapi_conn.pool
    logger.info(
        "Database connection pool status",
        size=pool.size(),
        checked_out=pool.checkedout(),
        overflow=pool.overflow()
    )
```

---

**Monitoring Dashboard**: [https://grafana.detect-project.com](https://grafana.detect-project.com)
**Logs**: [https://kibana.detect-project.com](https://kibana.detect-project.com)
**Traces**: [https://jaeger.detect-project.com](https://jaeger.detect-project.com)

**Last Updated**: 2025-01-15
