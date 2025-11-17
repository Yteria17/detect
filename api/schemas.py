"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class PriorityLevel(str, Enum):
    """Priority levels for fact-check requests."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class VerdictType(str, Enum):
    """Verdict types for fact-checking results."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    CONFLICTING = "CONFLICTING"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
    PENDING = "PENDING"


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    URGENT = "URGENT"


# Request Schemas


class FactCheckRequest(BaseModel):
    """Request schema for fact-checking a claim."""

    claim: str = Field(
        ..., min_length=10, max_length=5000, description="The claim to fact-check"
    )
    priority: PriorityLevel = Field(
        default=PriorityLevel.NORMAL, description="Priority level for processing"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata about the claim"
    )

    @validator("claim")
    def claim_not_empty(cls, v):
        """Validate claim is not just whitespace."""
        if not v.strip():
            raise ValueError("Claim cannot be empty or whitespace only")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "claim": "Jean Dupont, PDG de TechCorp, a déclaré que les ventes augmentent de 150%",
                "priority": "normal",
                "metadata": {"source": "twitter", "author": "@user123"},
            }
        }


class BulkFactCheckRequest(BaseModel):
    """Request schema for bulk fact-checking."""

    claims: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of claims to fact-check"
    )
    priority: PriorityLevel = Field(
        default=PriorityLevel.NORMAL, description="Priority level for all claims"
    )


# Response Schemas


class EvidenceSummary(BaseModel):
    """Summary of a piece of evidence."""

    source: str
    credibility: float = Field(ge=0.0, le=1.0)
    text_preview: str


class ConfidenceBreakdown(BaseModel):
    """Detailed confidence score breakdown."""

    overall: float = Field(ge=0.0, le=1.0)
    assertion_confidence: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    anomaly_penalty: float = Field(ge=0.0, le=1.0)


class FactCheckResponse(BaseModel):
    """Response schema for fact-check results."""

    claim_id: str
    status: str = Field(default="completed")
    original_claim: str
    verdict: VerdictType
    confidence: float = Field(ge=0.0, le=1.0)

    # Classification
    theme: str
    complexity: int = Field(ge=0, le=10)
    urgency: int = Field(ge=0, le=10)

    # Evidence
    evidence_count: int
    evidence_summary: List[EvidenceSummary]
    sources_used: List[str]
    high_credibility_sources: List[str]

    # Analysis
    key_findings: List[str]
    contradictions: List[str]
    recommended_actions: List[str]

    # Confidence
    confidence_breakdown: ConfidenceBreakdown

    # Metadata
    processing_time_seconds: float
    created_at: datetime
    updated_at: datetime

    class Config:
        schema_extra = {
            "example": {
                "claim_id": "claim_12345",
                "status": "completed",
                "original_claim": "Example claim",
                "verdict": "REFUTED",
                "confidence": 0.85,
                "theme": "politics",
                "complexity": 6,
                "urgency": 7,
                "evidence_count": 12,
                "evidence_summary": [],
                "sources_used": ["bbc.com", "reuters.com"],
                "high_credibility_sources": ["bbc.com"],
                "key_findings": ["Finding 1", "Finding 2"],
                "contradictions": [],
                "recommended_actions": ["Publish correction"],
                "confidence_breakdown": {
                    "overall": 0.85,
                    "assertion_confidence": 0.9,
                    "evidence_quality": 0.88,
                    "anomaly_penalty": 0.1,
                },
                "processing_time_seconds": 23.5,
                "created_at": "2025-01-15T10:30:00",
                "updated_at": "2025-01-15T10:30:23",
            }
        }


class FactCheckStatusResponse(BaseModel):
    """Response schema for checking fact-check status."""

    claim_id: str
    status: str
    verdict: Optional[VerdictType] = None
    confidence: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None


class AlertResponse(BaseModel):
    """Response schema for alerts."""

    alert_id: str
    level: AlertLevel
    message: str
    claim_id: str
    verdict: VerdictType
    confidence: float
    stakeholders: List[str]
    created_at: datetime


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""

    status: str
    version: str
    timestamp: datetime
    services: Dict[str, bool]


class StatsResponse(BaseModel):
    """Response schema for system statistics."""

    total_fact_checks: int
    active_fact_checks: int
    avg_processing_time: float
    verdicts_breakdown: Dict[str, int]
    avg_confidence: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str
    detail: Optional[str] = None
    claim_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Database Models (for internal use)


class FactCheckRecord(BaseModel):
    """Database record for fact-check."""

    claim_id: str
    claim: str
    verdict: str
    confidence: float
    reasoning_trace: List[str]
    evidence_used: List[Dict[str, Any]]
    created_at: datetime
    agents_involved: List[str]
