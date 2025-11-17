"""Type definitions for the fact-checking system."""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class Verdict(str, Enum):
    """Fact-check verdict enum."""
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    CONFLICTING = "CONFLICTING"
    PENDING = "PENDING"


class Priority(str, Enum):
    """Claim priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Source(BaseModel):
    """Evidence source information."""
    url: str
    domain: str
    title: Optional[str] = None
    credibility_score: float = Field(ge=0.0, le=1.0)
    timestamp: Optional[datetime] = None


class Evidence(BaseModel):
    """Evidence piece for fact-checking."""
    text: str
    source: Source
    relevance_score: float = Field(ge=0.0, le=1.0)
    assertion: Optional[str] = None


class Assertion(BaseModel):
    """Individual assertion extracted from a claim."""
    text: str
    assertion_id: str
    verdict: Verdict = Verdict.PENDING
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)


class Classification(BaseModel):
    """Claim classification result."""
    theme: str
    complexity: int = Field(ge=1, le=10)
    urgency: int = Field(ge=1, le=10)
    requires_multimodal: bool = False
    detected_entities: List[str] = Field(default_factory=list)


class AnomalyScore(BaseModel):
    """Anomaly detection result for an assertion."""
    assertion_id: str
    score: float = Field(ge=0.0, le=1.0)
    detected_patterns: List[str] = Field(default_factory=list)
    reasoning: str


class FactCheckingState(BaseModel):
    """Central state shared across all agents."""

    # Input
    claim_id: str
    original_claim: str
    priority: Priority = Priority.NORMAL

    # Intermediate results
    decomposed_assertions: List[Assertion] = Field(default_factory=list)
    classification: Optional[Classification] = None
    evidence_retrieved: List[Evidence] = Field(default_factory=list)
    anomaly_scores: Dict[str, AnomalyScore] = Field(default_factory=dict)
    triplet_verdicts: Dict[str, Dict] = Field(default_factory=dict)

    # Final results
    final_verdict: Verdict = Verdict.PENDING
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_trace: List[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    agents_involved: List[str] = Field(default_factory=list)
    processing_time_ms: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class ClaimRequest(BaseModel):
    """API request for fact-checking."""
    claim: str = Field(..., min_length=10, max_length=5000)
    priority: Priority = Priority.NORMAL
    source_url: Optional[str] = None
    metadata: Optional[Dict] = None


class ClaimResponse(BaseModel):
    """API response for fact-checking."""
    claim_id: str
    verdict: Verdict
    confidence: float
    created_at: datetime
    final_explanation: Optional[str] = None
    evidence_summary: Optional[List[Dict]] = None
    reasoning_trace: Optional[List[str]] = None
