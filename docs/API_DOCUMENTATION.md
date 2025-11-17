# API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Error Handling](#error-handling)
5. [Endpoints](#endpoints)
6. [WebSocket API](#websocket-api)
7. [Webhook Integration](#webhook-integration)
8. [SDKs & Client Libraries](#sdks--client-libraries)
9. [Examples](#examples)

---

## Overview

The Disinformation Detection API provides RESTful endpoints for fact-checking, batch processing, and monitoring. All endpoints return JSON responses and use standard HTTP status codes.

**Base URL**: `https://api.detect-project.com/v1`

**API Version**: v1.0

**OpenAPI Spec**: [https://api.detect-project.com/v1/openapi.json](https://api.detect-project.com/v1/openapi.json)

---

## Authentication

### API Keys

All API requests require authentication via API keys.

```bash
curl -X POST https://api.detect-project.com/v1/fact-check \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"claim": "Your claim here"}'
```

### Obtaining API Keys

```bash
# Register for an account
curl -X POST https://api.detect-project.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password",
    "organization": "Your Org"
  }'

# Response
{
  "api_key": "sk-proj-abc123...",
  "user_id": "user_uuid",
  "created_at": "2025-01-15T10:00:00Z"
}
```

### JWT Tokens (OAuth 2.0)

For web applications, use OAuth 2.0 flow:

```bash
# Step 1: Obtain authorization code
GET https://api.detect-project.com/v1/oauth/authorize?
    client_id=YOUR_CLIENT_ID&
    redirect_uri=https://yourapp.com/callback&
    response_type=code&
    scope=fact_check:read fact_check:write

# Step 2: Exchange code for token
POST https://api.detect-project.com/v1/oauth/token
{
  "grant_type": "authorization_code",
  "code": "AUTHORIZATION_CODE",
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET"
}

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_abc123..."
}
```

---

## Rate Limiting

Rate limits are applied per API key:

| Tier | Requests/Minute | Requests/Day | Concurrent Requests |
|------|-----------------|--------------|---------------------|
| Free | 10 | 1,000 | 2 |
| Pro | 100 | 50,000 | 10 |
| Enterprise | 1,000 | Unlimited | 50 |

### Rate Limit Headers

Every response includes rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1674216000
```

### Handling Rate Limits

```python
import time
import requests

def check_claim_with_retry(claim):
    while True:
        response = requests.post(
            "https://api.detect-project.com/v1/fact-check",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"claim": claim}
        )

        if response.status_code == 429:
            # Rate limited
            reset_time = int(response.headers['X-RateLimit-Reset'])
            wait_seconds = reset_time - time.time()
            print(f"Rate limited. Waiting {wait_seconds}s")
            time.sleep(wait_seconds + 1)
            continue

        return response.json()
```

---

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "INVALID_CLAIM",
    "message": "Claim text cannot be empty",
    "details": {
      "field": "claim",
      "constraint": "minLength"
    },
    "request_id": "req_abc123",
    "timestamp": "2025-01-15T10:00:00Z"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid input parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Temporary service outage |

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_CLAIM` | Claim format is invalid |
| `CLAIM_TOO_LONG` | Claim exceeds max length (5000 chars) |
| `CLAIM_TOO_SHORT` | Claim too short (< 10 chars) |
| `PROCESSING_FAILED` | Internal processing error |
| `INSUFFICIENT_EVIDENCE` | Not enough evidence to verify |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INVALID_API_KEY` | API key is invalid or expired |
| `QUOTA_EXCEEDED` | Monthly quota exceeded |

---

## Endpoints

### POST /fact-check

Submit a claim for fact-checking.

**Request**:
```http
POST /v1/fact-check
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "claim": "Jean Dupont, CEO of TechCorp, announced 150% revenue growth in 2024",
  "priority": "normal",
  "context": {
    "source_url": "https://example.com/article",
    "published_at": "2025-01-15T10:00:00Z"
  },
  "options": {
    "include_evidence": true,
    "include_reasoning": true,
    "deepfake_check": false,
    "language": "en"
  }
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `claim` | string | Yes | The claim to verify (10-5000 chars) |
| `priority` | string | No | Processing priority: `low`, `normal`, `high` (default: `normal`) |
| `context` | object | No | Additional context about the claim |
| `context.source_url` | string | No | Original source URL |
| `context.published_at` | datetime | No | Publication timestamp |
| `options` | object | No | Processing options |
| `options.include_evidence` | boolean | No | Include evidence in response (default: `true`) |
| `options.include_reasoning` | boolean | No | Include reasoning trace (default: `true`) |
| `options.deepfake_check` | boolean | No | Enable deepfake detection for media (default: `false`) |
| `options.language` | string | No | Language code (default: `en`) |

**Response** (200 OK):
```json
{
  "claim_id": "claim_abc123",
  "status": "completed",
  "verdict": "REFUTED",
  "confidence": 0.87,
  "created_at": "2025-01-15T10:00:00Z",
  "completed_at": "2025-01-15T10:00:24Z",
  "processing_time_ms": 24500,
  "classification": {
    "theme": "business",
    "complexity": 6,
    "urgency": 5
  },
  "assertions": [
    {
      "text": "Jean Dupont is CEO of TechCorp",
      "verdict": "SUPPORTED",
      "confidence": 0.95
    },
    {
      "text": "TechCorp revenue growth was 150% in 2024",
      "verdict": "REFUTED",
      "confidence": 0.82
    }
  ],
  "evidence": [
    {
      "source": "techcorp.com/investors",
      "text": "TechCorp reported 47% revenue growth in fiscal year 2024",
      "credibility": 0.95,
      "relevance": 0.98,
      "url": "https://techcorp.com/investors/reports/2024"
    },
    {
      "source": "reuters.com",
      "text": "Industry analysts confirm TechCorp's growth at 45-50%",
      "credibility": 0.92,
      "relevance": 0.85,
      "url": "https://reuters.com/article/techcorp-earnings"
    }
  ],
  "reasoning_trace": [
    "Classifier: Decomposed claim into 2 assertions",
    "Anomaly Detector: No suspicious patterns detected",
    "Fact-Checker: Retrieved 5 evidence sources",
    "Fact-Checker: Revenue claim contradicted by official filings",
    "Reporter: Final verdict REFUTED with high confidence"
  ],
  "alerts": [
    {
      "severity": "MEDIUM",
      "message": "Claim significantly inflates actual revenue figures",
      "recommendation": "Flag for correction with source"
    }
  ]
}
```

---

### GET /fact-check/{claim_id}

Retrieve results of a previous fact-check.

**Request**:
```http
GET /v1/fact-check/claim_abc123
Authorization: Bearer YOUR_API_KEY
```

**Response** (200 OK):
Same format as POST /fact-check response.

**Response** (404 Not Found):
```json
{
  "error": {
    "code": "CLAIM_NOT_FOUND",
    "message": "No fact-check found with ID claim_abc123"
  }
}
```

---

### POST /batch-check

Submit multiple claims for batch processing.

**Request**:
```http
POST /v1/batch-check
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "claims": [
    {
      "id": "custom_id_1",
      "claim": "First claim to check"
    },
    {
      "id": "custom_id_2",
      "claim": "Second claim to check"
    }
  ],
  "callback_url": "https://yourapp.com/webhook/batch-complete",
  "priority": "normal"
}
```

**Response** (202 Accepted):
```json
{
  "batch_id": "batch_xyz789",
  "status": "processing",
  "total_claims": 2,
  "estimated_completion": "2025-01-15T10:05:00Z",
  "progress_url": "/v1/batch-check/batch_xyz789"
}
```

---

### GET /batch-check/{batch_id}

Check batch processing status.

**Request**:
```http
GET /v1/batch-check/batch_xyz789
Authorization: Bearer YOUR_API_KEY
```

**Response** (200 OK):
```json
{
  "batch_id": "batch_xyz789",
  "status": "completed",
  "total_claims": 2,
  "completed": 2,
  "failed": 0,
  "progress": 100,
  "started_at": "2025-01-15T10:00:00Z",
  "completed_at": "2025-01-15T10:04:30Z",
  "results": [
    {
      "id": "custom_id_1",
      "claim_id": "claim_abc123",
      "verdict": "REFUTED",
      "confidence": 0.87
    },
    {
      "id": "custom_id_2",
      "claim_id": "claim_def456",
      "verdict": "SUPPORTED",
      "confidence": 0.92
    }
  ],
  "download_url": "/v1/batch-check/batch_xyz789/download"
}
```

---

### GET /batch-check/{batch_id}/download

Download batch results as CSV or JSON.

**Request**:
```http
GET /v1/batch-check/batch_xyz789/download?format=csv
Authorization: Bearer YOUR_API_KEY
```

**Parameters**:
- `format`: `json` or `csv` (default: `json`)

**Response** (200 OK):
Returns file download with appropriate Content-Type.

---

### POST /deepfake-detect

Analyze media for deepfake manipulation.

**Request**:
```http
POST /v1/deepfake-detect
Content-Type: multipart/form-data
Authorization: Bearer YOUR_API_KEY

file: <binary video/audio/image file>
type: "video"
options: {"check_audio": true, "check_video": true}
```

**Response** (200 OK):
```json
{
  "detection_id": "detect_123",
  "file_type": "video",
  "duration_seconds": 45.2,
  "analysis": {
    "overall_score": 0.78,
    "verdict": "LIKELY_DEEPFAKE",
    "confidence": 0.82
  },
  "audio_analysis": {
    "deepfake_probability": 0.75,
    "artifacts_detected": ["pitch_inconsistency", "spectral_anomalies"],
    "natural_prosody_score": 0.3
  },
  "video_analysis": {
    "deepfake_probability": 0.81,
    "biological_signals": {
      "ppg_detected": false,
      "blink_rate_normal": false
    },
    "face_manipulation_score": 0.85
  },
  "multimodal_consistency": {
    "lip_sync_score": 0.45,
    "audio_video_alignment": "POOR"
  },
  "recommendations": [
    "High likelihood of audio manipulation",
    "No biological signals detected in video",
    "Poor lip-sync suggests separate audio track"
  ]
}
```

---

### GET /sources/credibility/{domain}

Get credibility score for a source domain.

**Request**:
```http
GET /v1/sources/credibility/bbc.com
Authorization: Bearer YOUR_API_KEY
```

**Response** (200 OK):
```json
{
  "domain": "bbc.com",
  "credibility_score": 0.95,
  "category": "mainstream_media",
  "last_verified": "2025-01-15T00:00:00Z",
  "metrics": {
    "fact_check_accuracy": 0.97,
    "editorial_standards": 0.98,
    "transparency": 0.92,
    "correction_policy": true
  },
  "assessments": [
    {
      "organization": "NewsGuard",
      "score": 95,
      "date": "2024-12-01"
    },
    {
      "organization": "MBFC",
      "rating": "Least Biased",
      "date": "2024-11-15"
    }
  ]
}
```

---

### GET /health

Health check endpoint.

**Request**:
```http
GET /v1/health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "vector_db": "healthy",
    "llm_service": "healthy"
  },
  "metrics": {
    "uptime_seconds": 86400,
    "requests_last_minute": 45,
    "average_latency_ms": 24.5
  }
}
```

---

### GET /stats

Get platform statistics.

**Request**:
```http
GET /v1/stats
Authorization: Bearer YOUR_API_KEY
```

**Response** (200 OK):
```json
{
  "total_claims_checked": 1234567,
  "claims_today": 5432,
  "average_processing_time_ms": 24500,
  "verdicts_distribution": {
    "SUPPORTED": 0.35,
    "REFUTED": 0.45,
    "INSUFFICIENT_INFO": 0.20
  },
  "top_themes": [
    {"theme": "politics", "count": 45000},
    {"theme": "health", "count": 32000},
    {"theme": "business", "count": 28000}
  ],
  "performance": {
    "accuracy": 0.923,
    "f1_score": 0.87,
    "false_positive_rate": 0.042
  }
}
```

---

## WebSocket API

Real-time fact-checking updates via WebSocket.

### Connection

```javascript
const ws = new WebSocket('wss://api.detect-project.com/v1/ws');

// Authenticate
ws.send(JSON.stringify({
  type: 'auth',
  api_key: 'YOUR_API_KEY'
}));

// Subscribe to claim updates
ws.send(JSON.stringify({
  type: 'subscribe',
  claim_id: 'claim_abc123'
}));

// Receive updates
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Update:', update);
};
```

### Message Types

**Status Update**:
```json
{
  "type": "status_update",
  "claim_id": "claim_abc123",
  "status": "processing",
  "current_agent": "fact_checker",
  "progress": 60
}
```

**Result Available**:
```json
{
  "type": "result",
  "claim_id": "claim_abc123",
  "verdict": "REFUTED",
  "confidence": 0.87,
  "url": "/v1/fact-check/claim_abc123"
}
```

---

## Webhook Integration

Configure webhooks to receive notifications.

### Register Webhook

```http
POST /v1/webhooks
Authorization: Bearer YOUR_API_KEY

{
  "url": "https://yourapp.com/webhook/fact-check",
  "events": ["fact_check.completed", "fact_check.failed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload

```json
{
  "event": "fact_check.completed",
  "timestamp": "2025-01-15T10:00:24Z",
  "data": {
    "claim_id": "claim_abc123",
    "verdict": "REFUTED",
    "confidence": 0.87,
    "url": "/v1/fact-check/claim_abc123"
  }
}
```

### Verifying Webhooks

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    """Verify webhook signature"""
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected)
```

---

## SDKs & Client Libraries

### Python SDK

```bash
pip install detect-sdk
```

```python
from detect import DetectClient

client = DetectClient(api_key="YOUR_API_KEY")

# Single fact-check
result = client.fact_check("Your claim here")
print(f"Verdict: {result.verdict}")

# Batch processing
batch = client.batch_check([
    "First claim",
    "Second claim",
    "Third claim"
])
batch.wait()
print(f"Results: {batch.results}")

# Deepfake detection
deepfake = client.detect_deepfake("video.mp4")
print(f"Deepfake score: {deepfake.score}")
```

### JavaScript SDK

```bash
npm install @detect/sdk
```

```javascript
import { DetectClient } from '@detect/sdk';

const client = new DetectClient({ apiKey: 'YOUR_API_KEY' });

// Single fact-check
const result = await client.factCheck({
  claim: 'Your claim here'
});
console.log('Verdict:', result.verdict);

// With streaming updates
const stream = client.factCheckStream({
  claim: 'Your claim here'
});

stream.on('progress', (update) => {
  console.log('Progress:', update.progress);
});

stream.on('complete', (result) => {
  console.log('Complete:', result);
});
```

---

## Examples

### Complete Workflow Example

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.detect-project.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Submit claim
response = requests.post(
    f"{BASE_URL}/fact-check",
    headers=headers,
    json={
        "claim": "Jean Dupont announced 150% revenue growth",
        "options": {
            "include_evidence": True,
            "include_reasoning": True
        }
    }
)

result = response.json()
claim_id = result["claim_id"]
print(f"Submitted claim: {claim_id}")

# 2. Check status (if async)
if result["status"] == "processing":
    while True:
        status_response = requests.get(
            f"{BASE_URL}/fact-check/{claim_id}",
            headers=headers
        )
        status = status_response.json()

        if status["status"] == "completed":
            result = status
            break

        print(f"Progress: {status.get('progress', 0)}%")
        time.sleep(2)

# 3. Process results
print(f"\nVerdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}")
print(f"\nEvidence:")
for evidence in result['evidence'][:3]:
    print(f"  - {evidence['source']}: {evidence['text'][:100]}...")

print(f"\nReasoning:")
for step in result['reasoning_trace']:
    print(f"  {step}")
```

### Batch Processing Example

```python
# Prepare batch
claims = [
    {"id": "1", "claim": "First claim"},
    {"id": "2", "claim": "Second claim"},
    {"id": "3", "claim": "Third claim"}
]

# Submit batch
batch_response = requests.post(
    f"{BASE_URL}/batch-check",
    headers=headers,
    json={"claims": claims}
)

batch_id = batch_response.json()["batch_id"]

# Wait for completion
while True:
    status = requests.get(
        f"{BASE_URL}/batch-check/{batch_id}",
        headers=headers
    ).json()

    if status["status"] == "completed":
        break

    print(f"Progress: {status['progress']}%")
    time.sleep(5)

# Download results
results = requests.get(
    f"{BASE_URL}/batch-check/{batch_id}/download?format=json",
    headers=headers
).json()

for result in results:
    print(f"{result['id']}: {result['verdict']}")
```

---

## Support

- **API Status**: [https://status.detect-project.com](https://status.detect-project.com)
- **Documentation**: [https://docs.detect-project.com](https://docs.detect-project.com)
- **Support Email**: api-support@detect-project.com
- **Discord Community**: [https://discord.gg/detect](https://discord.gg/detect)

---

**Last Updated**: 2025-01-15

**API Version**: v1.0
