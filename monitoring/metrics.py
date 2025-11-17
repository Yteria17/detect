"""Prometheus metrics for monitoring system performance."""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable, Any


# Application Info
app_info = Info("factcheck_app", "Multi-Agent Fact-Checking System")

# Request Metrics
fact_check_requests_total = Counter(
    "fact_check_requests_total",
    "Total number of fact-check requests",
    ["verdict", "priority"],
)

fact_check_requests_failed = Counter(
    "fact_check_requests_failed",
    "Total number of failed fact-check requests",
    ["error_type"],
)

# Latency Metrics
fact_check_duration_seconds = Histogram(
    "fact_check_duration_seconds",
    "Time spent processing fact-check requests",
    ["agent", "complexity"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

agent_execution_duration_seconds = Histogram(
    "agent_execution_duration_seconds",
    "Time spent by individual agents",
    ["agent_name"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

# Agent Activity
active_fact_checks = Gauge(
    "active_fact_checks",
    "Number of fact-checks currently being processed",
)

agent_invocations_total = Counter(
    "agent_invocations_total",
    "Total number of agent invocations",
    ["agent_name", "status"],
)

# Evidence Retrieval
evidence_retrieved_total = Counter(
    "evidence_retrieved_total",
    "Total number of evidence pieces retrieved",
    ["source_type"],
)

evidence_credibility_score = Histogram(
    "evidence_credibility_score",
    "Credibility scores of retrieved evidence",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],
)

# Verdict Metrics
verdict_confidence_score = Histogram(
    "verdict_confidence_score",
    "Confidence scores of final verdicts",
    ["verdict"],
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],
)

# System Resources
vector_db_queries_total = Counter(
    "vector_db_queries_total",
    "Total number of vector database queries",
)

llm_api_calls_total = Counter(
    "llm_api_calls_total",
    "Total number of LLM API calls",
    ["model", "provider"],
)

llm_tokens_used_total = Counter(
    "llm_tokens_used_total",
    "Total number of LLM tokens used",
    ["model", "type"],  # type: input/output
)


def track_agent_execution(agent_name: str) -> Callable:
    """Decorator to track agent execution time and status.

    Args:
        agent_name: Name of the agent being tracked

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                agent_execution_duration_seconds.labels(
                    agent_name=agent_name
                ).observe(duration)
                agent_invocations_total.labels(
                    agent_name=agent_name, status=status
                ).inc()

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                agent_execution_duration_seconds.labels(
                    agent_name=agent_name
                ).observe(duration)
                agent_invocations_total.labels(
                    agent_name=agent_name, status=status
                ).inc()

        # Return appropriate wrapper based on function type
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x100:
            return async_wrapper
        return sync_wrapper

    return decorator


def track_fact_check_request(func: Callable) -> Callable:
    """Decorator to track complete fact-check requests.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        active_fact_checks.inc()
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)

            # Track metrics based on result
            if result and "verdict" in result:
                fact_check_requests_total.labels(
                    verdict=result["verdict"],
                    priority=kwargs.get("priority", "normal"),
                ).inc()

                if "confidence" in result:
                    verdict_confidence_score.labels(
                        verdict=result["verdict"]
                    ).observe(result["confidence"])

                complexity = result.get("classification", {}).get("complexity", "unknown")
                duration = time.time() - start_time
                fact_check_duration_seconds.labels(
                    agent="complete_pipeline", complexity=str(complexity)
                ).observe(duration)

            return result

        except Exception as e:
            fact_check_requests_failed.labels(
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            active_fact_checks.dec()

    return wrapper


# Initialize app info
app_info.info(
    {
        "version": "1.0.0",
        "environment": "production",
    }
)
