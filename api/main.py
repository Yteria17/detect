"""FastAPI application for Multi-Agent Fact-Checking System."""

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime
import asyncio
from typing import Dict, List
import uuid

from api.schemas import (
    FactCheckRequest,
    FactCheckResponse,
    FactCheckStatusResponse,
    HealthCheckResponse,
    StatsResponse,
    ErrorResponse,
    VerdictType,
    EvidenceSummary,
    ConfidenceBreakdown,
)
from agents.reporter import ReporterAgent
from config.settings import settings
from monitoring.logger import setup_logging, get_logger
from monitoring.metrics import (
    track_fact_check_request,
    fact_check_requests_total,
    active_fact_checks,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# In-memory storage (replace with database in production)
fact_check_results: Dict[str, Dict] = {}
fact_check_status: Dict[str, str] = {}

# Startup time for uptime calculation
startup_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    logger.info("application_starting", version=settings.APP_VERSION)

    # Initialize agents and services here
    # e.g., database connections, vector stores, etc.

    yield

    # Cleanup on shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-Agent System for Automatic Disinformation Detection",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
if settings.ENABLE_PROMETHEUS:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# Initialize agents
reporter_agent = ReporterAgent()


# Helper functions


async def run_fact_check_pipeline(claim_id: str, claim: str, priority: str):
    """Execute the complete fact-checking pipeline.

    This is a simplified version - in production, this would orchestrate
    the full LangGraph workflow with all 5 agents.
    """
    try:
        logger.info("fact_check_started", claim_id=claim_id, claim=claim[:100])

        fact_check_status[claim_id] = "processing"

        # Simulate multi-agent workflow
        # In production, this would call the LangGraph orchestrator
        await asyncio.sleep(2)  # Simulate processing

        # Mock state (replace with actual LangGraph execution)
        mock_state = {
            "claim_id": claim_id,
            "original_claim": claim,
            "decomposed_assertions": [
                "Assertion 1 from claim",
                "Assertion 2 from claim",
            ],
            "classification": {
                "theme": "politics",
                "complexity": 6,
                "urgency": 7,
            },
            "evidence_retrieved": [
                {
                    "source": "bbc.com",
                    "credibility": 0.95,
                    "text": "Supporting evidence from BBC...",
                    "assertion": "Assertion 1 from claim",
                },
                {
                    "source": "reuters.com",
                    "credibility": 0.95,
                    "text": "More evidence from Reuters...",
                    "assertion": "Assertion 2 from claim",
                },
            ],
            "anomaly_scores": {
                "Assertion 1 from claim": 0.2,
                "Assertion 2 from claim": 0.3,
            },
            "triplet_verdicts": {
                "Assertion 1 from claim": {
                    "verdict": "SUPPORTED",
                    "confidence": 0.85,
                },
                "Assertion 2 from claim": {
                    "verdict": "REFUTED",
                    "confidence": 0.90,
                },
            },
            "final_verdict": "REFUTED",
            "confidence": 0.85,
            "reasoning_trace": [
                "Classifier: Decomposed into 2 assertions",
                "Retriever: Retrieved 2 evidence pieces",
                "FactChecker: Verified 2 assertions",
                "Reporter: Final verdict REFUTED with confidence 0.85",
            ],
            "agents_involved": [
                "classifier",
                "retriever",
                "anomaly_detector",
                "fact_checker",
                "reporter",
            ],
            "processing_time": 2.5,
            "created_at": datetime.now().isoformat(),
        }

        # Generate report using Agent 5
        report = reporter_agent.generate_report(mock_state)

        # Generate alert if needed
        alert = reporter_agent.generate_alert(report)
        if alert:
            logger.warning(
                "alert_triggered",
                claim_id=claim_id,
                level=alert.level.value,
            )

        # Store result
        fact_check_results[claim_id] = report.to_dict()
        fact_check_status[claim_id] = "completed"

        logger.info(
            "fact_check_completed",
            claim_id=claim_id,
            verdict=report.verdict.value,
            confidence=report.confidence,
        )

    except Exception as e:
        logger.error(
            "fact_check_failed",
            claim_id=claim_id,
            error=str(e),
            exc_info=True,
        )
        fact_check_status[claim_id] = "failed"
        fact_check_results[claim_id] = {
            "error": str(e),
            "status": "failed",
        }


# API Endpoints


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with basic health check."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now(),
        services={
            "api": True,
            "database": True,  # Would check actual DB connection
            "vector_db": True,  # Would check actual Vector DB
        },
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now(),
        services={
            "api": True,
            "database": True,
            "vector_db": True,
            "redis": True,
        },
    )


@app.post(
    f"{settings.API_PREFIX}/fact-check",
    response_model=FactCheckStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_fact_check(
    request: FactCheckRequest, background_tasks: BackgroundTasks
):
    """Submit a claim for fact-checking.

    The fact-check will be processed asynchronously.
    Use the returned claim_id to check status and retrieve results.
    """
    claim_id = f"claim_{uuid.uuid4().hex[:12]}"

    # Track metrics
    fact_check_requests_total.labels(
        verdict="PENDING", priority=request.priority.value
    ).inc()

    # Start background processing
    background_tasks.add_task(
        run_fact_check_pipeline,
        claim_id,
        request.claim,
        request.priority.value,
    )

    fact_check_status[claim_id] = "pending"

    logger.info(
        "fact_check_submitted",
        claim_id=claim_id,
        priority=request.priority.value,
    )

    return FactCheckStatusResponse(
        claim_id=claim_id,
        status="pending",
        verdict=None,
        confidence=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        estimated_completion=None,
    )


@app.get(
    f"{settings.API_PREFIX}/fact-check/{{claim_id}}",
    response_model=FactCheckResponse,
)
async def get_fact_check_result(claim_id: str):
    """Get the result of a fact-check by claim_id."""
    if claim_id not in fact_check_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fact-check with ID {claim_id} not found",
        )

    status_value = fact_check_status[claim_id]

    if status_value == "pending" or status_value == "processing":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Fact-check is still {status_value}. Please try again later.",
        )

    if status_value == "failed":
        error_info = fact_check_results.get(claim_id, {})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fact-check failed: {error_info.get('error', 'Unknown error')}",
        )

    # Return completed result
    result = fact_check_results.get(claim_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found",
        )

    # Convert to response schema
    return FactCheckResponse(
        claim_id=result["claim_id"],
        status="completed",
        original_claim=result["original_claim"],
        verdict=VerdictType(result["verdict"]),
        confidence=result["confidence"],
        theme=result["theme"],
        complexity=result["complexity"],
        urgency=result["urgency"],
        evidence_count=result["evidence_count"],
        evidence_summary=[
            EvidenceSummary(**e) for e in result["evidence_summary"]
        ],
        sources_used=result["sources_used"],
        high_credibility_sources=result["high_credibility_sources"],
        key_findings=result["key_findings"],
        contradictions=result["contradictions"],
        recommended_actions=result["recommended_actions"],
        confidence_breakdown=ConfidenceBreakdown(**result["confidence_breakdown"]),
        processing_time_seconds=result["processing_time_seconds"],
        created_at=datetime.fromisoformat(result["created_at"]),
        updated_at=datetime.fromisoformat(result["updated_at"]),
    )


@app.get(
    f"{settings.API_PREFIX}/fact-check/{{claim_id}}/status",
    response_model=FactCheckStatusResponse,
)
async def get_fact_check_status(claim_id: str):
    """Get the status of a fact-check without full results."""
    if claim_id not in fact_check_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fact-check with ID {claim_id} not found",
        )

    status_value = fact_check_status[claim_id]
    result = fact_check_results.get(claim_id, {})

    return FactCheckStatusResponse(
        claim_id=claim_id,
        status=status_value,
        verdict=VerdictType(result["verdict"]) if "verdict" in result else None,
        confidence=result.get("confidence"),
        created_at=datetime.fromisoformat(
            result.get("created_at", datetime.now().isoformat())
        ),
        updated_at=datetime.fromisoformat(
            result.get("updated_at", datetime.now().isoformat())
        ),
    )


@app.get(f"{settings.API_PREFIX}/stats", response_model=StatsResponse)
async def get_statistics():
    """Get system statistics."""
    total_checks = len(fact_check_results)
    active_checks = sum(
        1 for s in fact_check_status.values() if s in ["pending", "processing"]
    )

    completed_results = [
        r for r in fact_check_results.values() if "verdict" in r
    ]

    verdicts_breakdown = {}
    total_confidence = 0
    total_processing_time = 0

    for result in completed_results:
        verdict = result.get("verdict", "UNKNOWN")
        verdicts_breakdown[verdict] = verdicts_breakdown.get(verdict, 0) + 1
        total_confidence += result.get("confidence", 0)
        total_processing_time += result.get("processing_time_seconds", 0)

    avg_confidence = (
        total_confidence / len(completed_results) if completed_results else 0
    )
    avg_processing_time = (
        total_processing_time / len(completed_results) if completed_results else 0
    )

    uptime = (datetime.now() - startup_time).total_seconds()

    return StatsResponse(
        total_fact_checks=total_checks,
        active_fact_checks=active_checks,
        avg_processing_time=avg_processing_time,
        verdicts_breakdown=verdicts_breakdown,
        avg_confidence=avg_confidence,
        uptime_seconds=uptime,
    )


@app.delete(f"{settings.API_PREFIX}/fact-check/{{claim_id}}")
async def delete_fact_check(claim_id: str):
    """Delete a fact-check result."""
    if claim_id not in fact_check_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fact-check with ID {claim_id} not found",
        )

    del fact_check_status[claim_id]
    if claim_id in fact_check_results:
        del fact_check_results[claim_id]

    return {"message": f"Fact-check {claim_id} deleted successfully"}


# Error handlers


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("unhandled_exception", exc_info=True, path=request.url.path)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else None,
        ).dict(),
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
