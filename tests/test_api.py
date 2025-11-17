"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from api.main import app
from api.schemas import FactCheckRequest, PriorityLevel


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_health_endpoint(self, client):
        """Test detailed health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["api"] is True


class TestFactCheckEndpoints:
    """Test fact-checking endpoints."""

    def test_create_fact_check(self, client):
        """Test creating a new fact-check request."""
        request_data = {
            "claim": "Jean Dupont is the CEO of TechCorp and sales increased by 150%",
            "priority": "normal",
        }

        response = client.post("/api/v1/fact-check", json=request_data)
        assert response.status_code == 202
        data = response.json()

        assert "claim_id" in data
        assert data["status"] in ["pending", "processing"]
        assert data["claim_id"].startswith("claim_")

    def test_create_fact_check_invalid_claim(self, client):
        """Test creating fact-check with invalid claim."""
        request_data = {
            "claim": "   ",  # Empty/whitespace claim
            "priority": "normal",
        }

        response = client.post("/api/v1/fact-check", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_create_fact_check_short_claim(self, client):
        """Test creating fact-check with too short claim."""
        request_data = {
            "claim": "Short",  # Too short
            "priority": "normal",
        }

        response = client.post("/api/v1/fact-check", json=request_data)
        assert response.status_code == 422

    def test_get_fact_check_status(self, client):
        """Test getting fact-check status."""
        # First create a fact-check
        request_data = {
            "claim": "Test claim for status check",
            "priority": "normal",
        }

        create_response = client.post("/api/v1/fact-check", json=request_data)
        claim_id = create_response.json()["claim_id"]

        # Get status
        status_response = client.get(f"/api/v1/fact-check/{claim_id}/status")
        assert status_response.status_code == 200
        data = status_response.json()

        assert data["claim_id"] == claim_id
        assert data["status"] in ["pending", "processing", "completed"]

    def test_get_nonexistent_fact_check(self, client):
        """Test getting a non-existent fact-check."""
        response = client.get("/api/v1/fact-check/nonexistent_id")
        assert response.status_code == 404

    def test_delete_fact_check(self, client):
        """Test deleting a fact-check."""
        # Create a fact-check first
        request_data = {
            "claim": "Test claim for deletion",
            "priority": "normal",
        }

        create_response = client.post("/api/v1/fact-check", json=request_data)
        claim_id = create_response.json()["claim_id"]

        # Delete it
        delete_response = client.delete(f"/api/v1/fact-check/{claim_id}")
        assert delete_response.status_code == 200

        # Verify it's deleted
        get_response = client.get(f"/api/v1/fact-check/{claim_id}")
        assert get_response.status_code == 404


class TestStatisticsEndpoint:
    """Test statistics endpoint."""

    def test_get_statistics(self, client):
        """Test getting system statistics."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()

        assert "total_fact_checks" in data
        assert "active_fact_checks" in data
        assert "avg_processing_time" in data
        assert "verdicts_breakdown" in data
        assert "avg_confidence" in data
        assert "uptime_seconds" in data


class TestPriorityLevels:
    """Test different priority levels."""

    @pytest.mark.parametrize(
        "priority",
        ["low", "normal", "high", "urgent"],
    )
    def test_priority_levels(self, client, priority):
        """Test all priority levels are accepted."""
        request_data = {
            "claim": "Test claim with varying priority levels",
            "priority": priority,
        }

        response = client.post("/api/v1/fact-check", json=request_data)
        assert response.status_code == 202


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint_exists(self, client):
        """Test that metrics endpoint is available."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"fact_check" in response.content


@pytest.mark.asyncio
class TestAsyncBehavior:
    """Test asynchronous behavior of the API."""

    async def test_concurrent_fact_checks(self, client):
        """Test handling multiple concurrent fact-checks."""
        import asyncio

        async def create_fact_check(claim_text):
            request_data = {
                "claim": claim_text,
                "priority": "normal",
            }
            return client.post("/api/v1/fact-check", json=request_data)

        # Create multiple fact-checks concurrently
        tasks = [
            create_fact_check(f"Test claim number {i}")
            for i in range(5)
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 202
