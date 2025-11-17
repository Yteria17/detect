"""Main FastAPI application."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import asyncio
from datetime import datetime

from utils.types import ClaimRequest, ClaimResponse, FactCheckingState, Priority
from utils.logger import log
from agents.orchestrator import create_orchestrator
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="Misinformation Detection API",
    description="Multi-agent system for automated fact-checking and misinformation detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = None

# In-memory storage for results (replace with proper DB in production)
fact_check_results: Dict[str, FactCheckingState] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    global orchestrator

    log.info("Starting Misinformation Detection API")

    # TODO: Initialize LLM client
    # from langchain_anthropic import ChatAnthropic
    # llm_client = ChatAnthropic(api_key=settings.anthropic_api_key, model=settings.llm_model)

    # For now, initialize without LLM (will use heuristics)
    orchestrator = create_orchestrator(llm_client=None, retriever=None)

    log.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    log.info("Shutting down Misinformation Detection API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Misinformation Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "fact_check": "/api/v1/fact-check",
            "get_result": "/api/v1/fact-check/{claim_id}",
            "health": "/health",
            "workflow": "/api/v1/workflow"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "orchestrator_ready": orchestrator is not None
    }


@app.post("/api/v1/fact-check", response_model=ClaimResponse)
async def fact_check_claim(request: ClaimRequest, background_tasks: BackgroundTasks):
    """
    Submit a claim for fact-checking.

    Args:
        request: Claim request with text and metadata
        background_tasks: FastAPI background tasks

    Returns:
        Claim response with initial status
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        log.info(f"Received fact-check request: {request.claim[:100]}...")

        # Execute fact-checking asynchronously
        result = await orchestrator.check_claim(
            claim=request.claim,
            priority=request.priority
        )

        # Store result
        fact_check_results[result.claim_id] = result

        # Create response
        response = ClaimResponse(
            claim_id=result.claim_id,
            verdict=result.final_verdict,
            confidence=result.confidence,
            created_at=result.created_at,
            final_explanation=None,  # Can add summary here
            evidence_summary=[
                {
                    "domain": e.source.domain,
                    "credibility": e.source.credibility_score,
                    "url": e.source.url
                }
                for e in result.evidence_retrieved[:5]
            ] if result.evidence_retrieved else None,
            reasoning_trace=result.reasoning_trace
        )

        log.info(f"Fact-check completed: {result.claim_id} -> {result.final_verdict.value}")

        return response

    except Exception as e:
        log.error(f"Error processing fact-check request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/api/v1/fact-check/{claim_id}")
async def get_fact_check_result(claim_id: str):
    """
    Retrieve fact-check result by claim ID.

    Args:
        claim_id: Claim identifier

    Returns:
        Fact-check result
    """
    if claim_id not in fact_check_results:
        raise HTTPException(status_code=404, detail=f"Claim ID {claim_id} not found")

    result = fact_check_results[claim_id]

    # Generate full report
    from agents.reporter_agent import ReporterAgent
    reporter = ReporterAgent()
    report = reporter.generate_json_report(result)

    return report


@app.get("/api/v1/fact-check")
async def list_fact_checks(limit: int = 10, offset: int = 0):
    """
    List recent fact-check results.

    Args:
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        List of fact-check summaries
    """
    results = list(fact_check_results.values())

    # Sort by creation time (most recent first)
    results.sort(key=lambda r: r.created_at, reverse=True)

    # Paginate
    paginated = results[offset:offset + limit]

    return {
        "total": len(results),
        "limit": limit,
        "offset": offset,
        "results": [
            {
                "claim_id": r.claim_id,
                "claim": r.original_claim[:200],
                "verdict": r.final_verdict.value,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat(),
                "processing_time_ms": r.processing_time_ms
            }
            for r in paginated
        ]
    }


@app.delete("/api/v1/fact-check/{claim_id}")
async def delete_fact_check(claim_id: str):
    """
    Delete a fact-check result.

    Args:
        claim_id: Claim identifier

    Returns:
        Deletion confirmation
    """
    if claim_id not in fact_check_results:
        raise HTTPException(status_code=404, detail=f"Claim ID {claim_id} not found")

    del fact_check_results[claim_id]

    return {"message": f"Claim {claim_id} deleted successfully"}


@app.get("/api/v1/workflow")
async def get_workflow_info():
    """
    Get information about the fact-checking workflow.

    Returns:
        Workflow structure and agent information
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    return {
        "workflow_type": "Sequential Pipeline",
        "agents": [
            {
                "order": 1,
                "name": "ClassifierAgent",
                "description": "Categorizes claims and decomposes them into verifiable assertions"
            },
            {
                "order": 2,
                "name": "CollectorAgent",
                "description": "Monitors and collects content from various public sources"
            },
            {
                "order": 3,
                "name": "AnomalyDetectorAgent",
                "description": "Detects semantic anomalies and suspicious patterns in claims"
            },
            {
                "order": 4,
                "name": "FactCheckerAgent",
                "description": "Verifies claims using evidence and reasoning"
            },
            {
                "order": 5,
                "name": "ReporterAgent",
                "description": "Consolidates results and generates comprehensive reports"
            }
        ],
        "visualization": orchestrator.get_workflow_visualization()
    }


@app.get("/api/v1/stats")
async def get_statistics():
    """
    Get system statistics.

    Returns:
        Statistics about processed claims
    """
    if not fact_check_results:
        return {
            "total_claims": 0,
            "verdict_distribution": {},
            "average_confidence": 0.0,
            "average_processing_time_ms": 0
        }

    results = list(fact_check_results.values())

    # Verdict distribution
    verdict_counts = {}
    for result in results:
        verdict = result.final_verdict.value
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    # Average confidence
    avg_confidence = sum(r.confidence for r in results) / len(results)

    # Average processing time
    processing_times = [r.processing_time_ms for r in results if r.processing_time_ms]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

    return {
        "total_claims": len(results),
        "verdict_distribution": verdict_counts,
        "average_confidence": avg_confidence,
        "average_processing_time_ms": avg_time,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
