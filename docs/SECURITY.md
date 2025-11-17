# Security Guide

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Protection](#data-protection)
4. [API Security](#api-security)
5. [Input Validation](#input-validation)
6. [OWASP Top 10 Mitigation](#owasp-top-10-mitigation)
7. [Dependency Management](#dependency-management)
8. [Incident Response](#incident-response)
9. [Reporting Vulnerabilities](#reporting-vulnerabilities)

---

## Security Overview

### Security Principles

1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Minimum necessary access rights
3. **Fail Secure**: Default to secure state on failure
4. **Separation of Duties**: No single point of total control
5. **Audit Everything**: Comprehensive logging and monitoring

### Threat Model

| Threat | Risk Level | Mitigation |
|--------|-----------|------------|
| SQL Injection | HIGH | Parameterized queries, ORM |
| XSS | MEDIUM | Input sanitization, CSP headers |
| API Key Leakage | HIGH | Encryption, rotation, monitoring |
| DDoS | MEDIUM | Rate limiting, load balancing |
| Data Breach | HIGH | Encryption at rest/transit, access controls |
| Prompt Injection (LLM) | MEDIUM | Input validation, prompt hardening |

---

## Authentication & Authorization

### API Key Management

```python
# Generate secure API keys
import secrets
import hashlib

def generate_api_key() -> str:
    """Generate cryptographically secure API key"""
    random_bytes = secrets.token_bytes(32)
    key = f"sk-proj-{secrets.token_urlsafe(32)}"
    return key

def hash_api_key(key: str) -> str:
    """Hash API key for storage"""
    salt = secrets.token_bytes(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        key.encode(),
        salt,
        100000
    )
    return f"{salt.hex()}${hashed.hex()}"

def verify_api_key(key: str, stored_hash: str) -> bool:
    """Verify API key against stored hash"""
    salt_hex, hash_hex = stored_hash.split('$')
    salt = bytes.fromhex(salt_hex)

    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        key.encode(),
        salt,
        100000
    )

    return hashed.hex() == hash_hex
```

### JWT Implementation

```python
from datetime import datetime, timedelta
import jwt

SECRET_KEY = os.getenv('SECRET_KEY')  # 256-bit random key
ALGORITHM = "HS256"

def create_access_token(user_id: str, expires_delta: timedelta = None) -> str:
    """Create JWT access token"""
    to_encode = {"sub": user_id}

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Role-Based Access Control (RBAC)

```python
from enum import Enum
from functools import wraps

class Role(Enum):
    USER = "user"
    ADMIN = "admin"
    SUPERUSER = "superuser"

ROLE_PERMISSIONS = {
    Role.USER: ["fact_check:read", "fact_check:write"],
    Role.ADMIN: ["fact_check:*", "users:read", "sources:*"],
    Role.SUPERUSER: ["*"]
}

def require_permission(permission: str):
    """Decorator to enforce permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User, **kwargs):
            if not has_permission(current_user.role, permission):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

# Usage
@app.post("/admin/users")
@require_permission("users:write")
async def create_user(user_data: UserCreate, current_user: User):
    pass
```

---

## Data Protection

### Encryption at Rest

```python
from cryptography.fernet import Fernet

class DataEncryption:
    """Encrypt sensitive data at rest"""

    def __init__(self):
        # Load encryption key from secure vault
        self.key = os.getenv('ENCRYPTION_KEY').encode()
        self.fernet = Fernet(self.key)

    def encrypt(self, data: str) -> str:
        """Encrypt plaintext data"""
        encrypted = self.fernet.encrypt(data.encode())
        return encrypted.decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted data"""
        decrypted = self.fernet.decrypt(encrypted_data.encode())
        return decrypted.decode()

# Encrypt sensitive fields
encryptor = DataEncryption()

# Before storing in database
user.api_key = encryptor.encrypt(api_key)

# After retrieving
api_key = encryptor.decrypt(user.api_key)
```

### Encryption in Transit

```yaml
# nginx.conf - Force HTTPS

server {
    listen 80;
    server_name api.detect-project.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.detect-project.com;

    ssl_certificate /etc/letsencrypt/live/api.detect-project.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.detect-project.com/privkey.pem;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
    }
}
```

### Data Anonymization

```python
import hashlib

def anonymize_claim(claim: str, salt: str) -> str:
    """Anonymize claim for logging/analytics"""
    # Hash PII
    claim_anonymized = claim

    # Replace emails
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, claim)

    for email in emails:
        hashed = hashlib.sha256((email + salt).encode()).hexdigest()[:8]
        claim_anonymized = claim_anonymized.replace(email, f"[EMAIL:{hashed}]")

    # Replace phone numbers
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, claim)

    for phone in phones:
        hashed = hashlib.sha256((phone + salt).encode()).hexdigest()[:8]
        claim_anonymized = claim_anonymized.replace(phone, f"[PHONE:{hashed}]")

    return claim_anonymized
```

---

## API Security

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/fact-check")
@limiter.limit("10/minute")
async def fact_check(request: Request, claim: ClaimInput):
    """Rate-limited fact-check endpoint"""
    pass

# Custom rate limit by user tier
def rate_limit_by_tier(request: Request) -> str:
    """Dynamic rate limit based on user tier"""
    user = get_current_user(request)

    limits = {
        "free": "10/minute",
        "pro": "100/minute",
        "enterprise": "1000/minute"
    }

    return limits.get(user.tier, "10/minute")
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://detect-project.com",
        "https://dashboard.detect-project.com"
    ],  # Only allow specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Limit methods
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600
)
```

### Security Headers

```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # XSS protection
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:;"
        )

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response

app.add_middleware(SecurityHeadersMiddleware)
```

---

## Input Validation

### Claim Validation

```python
from pydantic import BaseModel, validator, constr
import re

class ClaimInput(BaseModel):
    """Validated claim input"""

    claim: constr(min_length=10, max_length=5000)
    source_url: Optional[HttpUrl] = None
    priority: str = "normal"

    @validator('claim')
    def sanitize_claim(cls, v):
        """Sanitize claim input"""
        # Remove null bytes
        v = v.replace('\x00', '')

        # Remove control characters except newline/tab
        v = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', v)

        # Limit consecutive whitespace
        v = re.sub(r'\s+', ' ', v)

        return v.strip()

    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority value"""
        allowed = ['low', 'normal', 'high']
        if v not in allowed:
            raise ValueError(f"Priority must be one of {allowed}")
        return v

    @validator('source_url')
    def validate_url(cls, v):
        """Validate URL is not malicious"""
        if v is None:
            return v

        # Blocklist check
        blocked_domains = ['malicious.com', 'phishing.net']
        if any(domain in str(v) for domain in blocked_domains):
            raise ValueError("URL from blocked domain")

        return v
```

### SQL Injection Prevention

```python
from sqlalchemy import text

# ❌ NEVER do this (vulnerable to SQL injection)
def get_claim_unsafe(claim_id: str):
    query = f"SELECT * FROM claims WHERE id = '{claim_id}'"
    return db.execute(query)

# ✅ Always use parameterized queries
def get_claim_safe(claim_id: str):
    query = text("SELECT * FROM claims WHERE id = :claim_id")
    return db.execute(query, {"claim_id": claim_id})

# ✅ Or use ORM
def get_claim_orm(claim_id: str):
    return db.query(FactCheck).filter(FactCheck.id == claim_id).first()
```

### LLM Prompt Injection Prevention

```python
def sanitize_prompt_input(user_input: str) -> str:
    """Prevent prompt injection attacks"""

    # Remove prompt manipulation attempts
    blocked_patterns = [
        r'ignore (previous|all) instructions',
        r'system:',
        r'</s>',
        r'<\|endoftext\|>',
    ]

    for pattern in blocked_patterns:
        user_input = re.sub(pattern, '[BLOCKED]', user_input, flags=re.IGNORECASE)

    # Escape special tokens
    user_input = user_input.replace('<', '&lt;').replace('>', '&gt;')

    return user_input

def build_safe_prompt(claim: str) -> str:
    """Build prompt with clear boundaries"""
    sanitized_claim = sanitize_prompt_input(claim)

    prompt = f"""
You are a fact-checking assistant. Analyze the following claim:

--- BEGIN CLAIM ---
{sanitized_claim}
--- END CLAIM ---

Respond only with your fact-check analysis in JSON format.
"""

    return prompt
```

---

## OWASP Top 10 Mitigation

### 1. Broken Access Control

```python
# Verify user owns resource before modification
@app.delete("/api/claims/{claim_id}")
async def delete_claim(claim_id: str, current_user: User):
    claim = db.get_claim(claim_id)

    if not claim:
        raise HTTPException(404, "Claim not found")

    # Check ownership
    if claim.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(403, "Not authorized to delete this claim")

    db.delete_claim(claim_id)
    return {"status": "deleted"}
```

### 2. Cryptographic Failures

```python
# Use strong, industry-standard algorithms
import bcrypt

def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode(), hashed.encode())
```

### 3. Injection

Covered in [Input Validation](#input-validation)

### 4. Insecure Design

```python
# Implement security by design
class SecureFactCheckPipeline:
    """Fact-checking pipeline with built-in security"""

    def __init__(self):
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()

    async def check_claim(self, claim: str, user: User) -> Result:
        """Secure fact-checking flow"""
        # 1. Rate limiting
        if not self.rate_limiter.check(user):
            raise RateLimitError("Too many requests")

        # 2. Input validation
        validated_claim = self.input_validator.validate(claim)

        # 3. Audit logging
        self.audit_logger.log("fact_check_started", {
            "user_id": user.id,
            "claim_preview": claim[:100]
        })

        # 4. Process
        try:
            result = await self.process(validated_claim)
        except Exception as e:
            self.audit_logger.log("fact_check_failed", {
                "user_id": user.id,
                "error": str(e)
            })
            raise

        # 5. Audit success
        self.audit_logger.log("fact_check_completed", {
            "user_id": user.id,
            "verdict": result.verdict
        })

        return result
```

### 5. Security Misconfiguration

```yaml
# .env.production - Secure configuration

# Disable debug mode
DEBUG=false

# Secure session settings
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=strict

# Strong secret keys (256-bit)
SECRET_KEY=<randomly-generated-256-bit-key>
ENCRYPTION_KEY=<randomly-generated-256-bit-key>

# Database security
DATABASE_SSL_MODE=require
DATABASE_POOL_MIN_SIZE=5
DATABASE_POOL_MAX_SIZE=20

# API security
ALLOWED_HOSTS=api.detect-project.com
CORS_ORIGINS=https://detect-project.com

# Logging
LOG_LEVEL=INFO
LOG_SENSITIVE_DATA=false
```

### 6-10: Additional Mitigations

See full [OWASP Top 10 Checklist](docs/security/owasp_checklist.md)

---

## Dependency Management

### Vulnerability Scanning

```bash
# Scan dependencies for vulnerabilities
pip-audit

# Update vulnerable dependencies
pip install --upgrade <package>

# Use safety for CI/CD
safety check --json
```

### Dependency Pinning

```txt
# requirements.txt - Pin exact versions

fastapi==0.109.0  # Not fastapi>=0.109.0
uvicorn==0.27.0
pydantic==2.5.3
```

### Regular Updates

```yaml
# .github/workflows/dependency-update.yml

name: Dependency Updates

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Update dependencies
        run: |
          pip-audit
          safety check
      - name: Create PR if vulnerabilities found
        # ... create automated PR
```

---

## Incident Response

### Security Incident Process

1. **Detection**: Monitoring alerts on suspicious activity
2. **Containment**: Isolate affected systems
3. **Investigation**: Determine scope and impact
4. **Eradication**: Remove threat and vulnerabilities
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Post-incident review

### Incident Response Checklist

- [ ] Isolate affected systems
- [ ] Preserve logs and evidence
- [ ] Notify security team
- [ ] Assess data breach scope
- [ ] Rotate compromised credentials
- [ ] Apply security patches
- [ ] Notify affected users (if required)
- [ ] Document incident timeline
- [ ] Conduct post-mortem
- [ ] Update security procedures

---

## Reporting Vulnerabilities

### Responsible Disclosure

If you discover a security vulnerability:

1. **DO NOT** disclose publicly
2. Email security@detect-project.com with:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

3. We will respond within **48 hours**
4. Fix will be developed and deployed
5. Public disclosure after fix (coordinated)

### Bug Bounty Program

- **Critical vulnerabilities**: $500-$2000
- **High severity**: $200-$500
- **Medium severity**: $50-$200
- **Low severity**: Recognition in security hall of fame

See [Bug Bounty Program](https://detect-project.com/security/bounty)

---

## Security Best Practices Summary

✅ **Always**:
- Use parameterized queries
- Validate and sanitize all inputs
- Encrypt sensitive data
- Use HTTPS everywhere
- Implement rate limiting
- Log security events
- Keep dependencies updated
- Use strong authentication

❌ **Never**:
- Store passwords in plaintext
- Trust user input
- Expose API keys in code
- Use weak encryption
- Disable security features
- Ignore security warnings

---

**Security Contact**: security@detect-project.com
**Last Security Audit**: 2025-01-15
**Next Scheduled Audit**: 2025-04-15
