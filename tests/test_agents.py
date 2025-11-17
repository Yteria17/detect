"""Tests for the multi-agent system."""

import pytest
from datetime import datetime
from agents.reporter import ReporterAgent, VerdictType, AlertLevel


@pytest.fixture
def reporter_agent():
    """Create a ReporterAgent instance."""
    return ReporterAgent()


@pytest.fixture
def sample_state():
    """Create a sample state for testing."""
    return {
        "claim_id": "test_claim_123",
        "original_claim": "Jean Dupont, CEO of TechCorp, claimed sales increased by 150%",
        "decomposed_assertions": [
            "Jean Dupont is CEO of TechCorp",
            "Sales increased by 150%",
        ],
        "classification": {
            "theme": "business",
            "complexity": 6,
            "urgency": 5,
        },
        "evidence_retrieved": [
            {
                "source": "reuters.com",
                "credibility": 0.95,
                "text": "Jean Dupont is indeed the CEO of TechCorp according to company records.",
                "assertion": "Jean Dupont is CEO of TechCorp",
            },
            {
                "source": "techcrunch.com",
                "credibility": 0.85,
                "text": "TechCorp reported 45% sales growth, not 150%.",
                "assertion": "Sales increased by 150%",
            },
        ],
        "anomaly_scores": {
            "Jean Dupont is CEO of TechCorp": 0.1,
            "Sales increased by 150%": 0.7,
        },
        "triplet_verdicts": {
            "Jean Dupont is CEO of TechCorp": {
                "verdict": "SUPPORTED",
                "confidence": 0.95,
            },
            "Sales increased by 150%": {
                "verdict": "REFUTED",
                "confidence": 0.85,
            },
        },
        "final_verdict": "REFUTED",
        "confidence": 0.85,
        "reasoning_trace": [
            "Classifier: Decomposed into 2 assertions",
            "Retriever: Retrieved 2 evidence pieces",
            "FactChecker: Verified 2 assertions",
            "Reporter: Final verdict REFUTED",
        ],
        "agents_involved": ["classifier", "retriever", "fact_checker", "reporter"],
        "processing_time": 3.5,
        "created_at": datetime.now().isoformat(),
    }


class TestReporterAgent:
    """Test the Reporter Agent functionality."""

    def test_generate_report(self, reporter_agent, sample_state):
        """Test report generation."""
        report = reporter_agent.generate_report(sample_state)

        assert report is not None
        assert report.claim_id == "test_claim_123"
        assert report.verdict == VerdictType.REFUTED
        assert 0 <= report.confidence <= 1
        assert report.theme == "business"
        assert report.evidence_count == 2
        assert len(report.sources_used) > 0

    def test_report_has_key_findings(self, reporter_agent, sample_state):
        """Test that report includes key findings."""
        report = reporter_agent.generate_report(sample_state)

        assert len(report.key_findings) > 0
        assert any("claim" in finding.lower() for finding in report.key_findings)

    def test_report_has_recommendations(self, reporter_agent, sample_state):
        """Test that report includes recommendations."""
        report = reporter_agent.generate_report(sample_state)

        assert len(report.recommended_actions) >= 0

    def test_confidence_breakdown(self, reporter_agent, sample_state):
        """Test confidence breakdown calculation."""
        report = reporter_agent.generate_report(sample_state)

        breakdown = report.confidence_breakdown
        assert "overall" in breakdown
        assert "assertion_confidence" in breakdown
        assert "evidence_quality" in breakdown
        assert "anomaly_penalty" in breakdown

        # All confidence values should be between 0 and 1
        for key, value in breakdown.items():
            assert 0 <= value <= 1

    def test_evidence_summary(self, reporter_agent, sample_state):
        """Test evidence summarization."""
        report = reporter_agent.generate_report(sample_state)

        assert len(report.evidence_summary) > 0
        for evidence in report.evidence_summary:
            assert "source" in evidence
            assert "credibility" in evidence
            assert "text_preview" in evidence

    def test_high_credibility_sources(self, reporter_agent, sample_state):
        """Test identification of high-credibility sources."""
        report = reporter_agent.generate_report(sample_state)

        # Reuters has high credibility (0.95)
        assert "reuters.com" in report.high_credibility_sources

    def test_generate_alert_for_high_confidence_refutation(
        self, reporter_agent, sample_state
    ):
        """Test alert generation for high-confidence refutation."""
        # Modify state for high-confidence refutation
        sample_state["confidence"] = 0.90
        sample_state["final_verdict"] = "REFUTED"

        report = reporter_agent.generate_report(sample_state)
        alert = reporter_agent.generate_alert(report)

        assert alert is not None
        assert alert.level in [AlertLevel.CRITICAL, AlertLevel.WARNING]
        assert alert.verdict == VerdictType.REFUTED

    def test_generate_alert_for_urgent_claim(self, reporter_agent, sample_state):
        """Test alert generation for urgent claims."""
        # Modify state for high urgency
        sample_state["classification"]["urgency"] = 9
        sample_state["confidence"] = 0.75

        report = reporter_agent.generate_report(sample_state)
        alert = reporter_agent.generate_alert(report)

        assert alert is not None
        assert alert.level == AlertLevel.URGENT

    def test_no_alert_for_low_confidence(self, reporter_agent, sample_state):
        """Test that no alert is generated for low confidence results."""
        # Modify state for low confidence
        sample_state["confidence"] = 0.45
        sample_state["final_verdict"] = "INSUFFICIENT_INFO"

        report = reporter_agent.generate_report(sample_state)
        alert = reporter_agent.generate_alert(report)

        # Should generate an INFO alert or no alert
        if alert:
            assert alert.level == AlertLevel.INFO

    def test_report_to_dict(self, reporter_agent, sample_state):
        """Test report conversion to dictionary."""
        report = reporter_agent.generate_report(sample_state)
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "claim_id" in report_dict
        assert "verdict" in report_dict
        assert "confidence" in report_dict
        assert "created_at" in report_dict

    def test_report_to_json(self, reporter_agent, sample_state):
        """Test report conversion to JSON."""
        import json

        report = reporter_agent.generate_report(sample_state)
        report_json = report.to_json()

        # Should be valid JSON
        parsed = json.loads(report_json)
        assert isinstance(parsed, dict)
        assert parsed["claim_id"] == "test_claim_123"

    def test_contradictions_detection(self, reporter_agent):
        """Test detection of contradictions in evidence."""
        conflicting_state = {
            "claim_id": "conflicting_claim",
            "original_claim": "Test conflicting claim",
            "decomposed_assertions": ["assertion1", "assertion2"],
            "classification": {"theme": "test", "complexity": 5, "urgency": 5},
            "evidence_retrieved": [],
            "anomaly_scores": {},
            "triplet_verdicts": {
                "assertion1": {"verdict": "SUPPORTED", "confidence": 0.8},
                "assertion2": {"verdict": "REFUTED", "confidence": 0.8},
            },
            "final_verdict": "CONFLICTING",
            "confidence": 0.6,
            "reasoning_trace": [],
            "agents_involved": [],
            "processing_time": 2.0,
            "created_at": datetime.now().isoformat(),
        }

        report = reporter_agent.generate_report(conflicting_state)

        # Should detect contradictions
        assert len(report.contradictions) > 0


class TestVerdictDetermination:
    """Test verdict determination logic."""

    def test_supported_verdict(self, reporter_agent):
        """Test fully supported verdict."""
        state = {
            "claim_id": "supported_claim",
            "original_claim": "Test claim",
            "decomposed_assertions": ["assertion1"],
            "classification": {"theme": "test", "complexity": 3, "urgency": 3},
            "evidence_retrieved": [
                {"source": "test.com", "credibility": 0.9, "text": "evidence"}
            ],
            "anomaly_scores": {"assertion1": 0.1},
            "triplet_verdicts": {
                "assertion1": {"verdict": "SUPPORTED", "confidence": 0.95}
            },
            "final_verdict": "SUPPORTED",
            "confidence": 0.95,
            "reasoning_trace": [],
            "agents_involved": [],
            "processing_time": 1.0,
            "created_at": datetime.now().isoformat(),
        }

        report = reporter_agent.generate_report(state)
        assert report.verdict == VerdictType.SUPPORTED

    def test_refuted_verdict(self, reporter_agent):
        """Test refuted verdict."""
        state = {
            "claim_id": "refuted_claim",
            "original_claim": "Test claim",
            "decomposed_assertions": ["assertion1"],
            "classification": {"theme": "test", "complexity": 3, "urgency": 3},
            "evidence_retrieved": [
                {"source": "test.com", "credibility": 0.9, "text": "evidence"}
            ],
            "anomaly_scores": {"assertion1": 0.2},
            "triplet_verdicts": {
                "assertion1": {"verdict": "REFUTED", "confidence": 0.9}
            },
            "final_verdict": "REFUTED",
            "confidence": 0.9,
            "reasoning_trace": [],
            "agents_involved": [],
            "processing_time": 1.0,
            "created_at": datetime.now().isoformat(),
        }

        report = reporter_agent.generate_report(state)
        assert report.verdict == VerdictType.REFUTED


class TestAlertStakeholders:
    """Test stakeholder determination for alerts."""

    def test_high_confidence_refutation_stakeholders(self, reporter_agent, sample_state):
        """Test stakeholders for high-confidence refutation."""
        sample_state["confidence"] = 0.90
        sample_state["final_verdict"] = "REFUTED"

        report = reporter_agent.generate_report(sample_state)
        alert = reporter_agent.generate_alert(report)

        if alert:
            assert "journalists" in alert.stakeholders or "public" in alert.stakeholders

    def test_urgent_claim_stakeholders(self, reporter_agent, sample_state):
        """Test stakeholders for urgent claims."""
        sample_state["classification"]["urgency"] = 9

        report = reporter_agent.generate_report(sample_state)
        alert = reporter_agent.generate_alert(report)

        if alert:
            assert "regulators" in alert.stakeholders
