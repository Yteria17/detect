"""Agent 5: Reporter - Structured reporting, alerts, and history tracking."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from dataclasses import dataclass, asdict

from monitoring.logger import get_logger
from monitoring.metrics import track_agent_execution

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    URGENT = "URGENT"


class VerdictType(str, Enum):
    """Verdict types for fact-checking results."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    CONFLICTING = "CONFLICTING"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


@dataclass
class Alert:
    """Alert data structure."""

    level: AlertLevel
    message: str
    claim_id: str
    verdict: VerdictType
    confidence: float
    reasoning: str
    created_at: datetime
    stakeholders: List[str]  # journalists, regulators, public
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["level"] = self.level.value
        data["verdict"] = self.verdict.value
        return data


@dataclass
class FactCheckReport:
    """Complete fact-check report structure."""

    claim_id: str
    original_claim: str
    verdict: VerdictType
    confidence: float

    # Classification
    theme: str
    complexity: int
    urgency: int

    # Evidence
    evidence_count: int
    evidence_summary: List[Dict[str, Any]]
    sources_used: List[str]
    high_credibility_sources: List[str]

    # Analysis
    decomposed_assertions: List[str]
    assertion_verdicts: Dict[str, Any]
    anomaly_scores: Dict[str, float]

    # Reasoning
    reasoning_trace: List[str]
    key_findings: List[str]
    contradictions: List[str]

    # Metadata
    agents_involved: List[str]
    processing_time_seconds: float
    created_at: datetime
    updated_at: datetime

    # Recommendations
    recommended_actions: List[str]
    confidence_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["verdict"] = self.verdict.value
        return data

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ReporterAgent:
    """Agent 5: Consolidates results, generates reports, and manages alerts."""

    def __init__(self):
        """Initialize the Reporter Agent."""
        self.logger = logger
        self.alert_thresholds = {
            AlertLevel.INFO: 0.0,
            AlertLevel.WARNING: 0.6,
            AlertLevel.CRITICAL: 0.8,
            AlertLevel.URGENT: 0.95,
        }

    @track_agent_execution("reporter")
    def generate_report(self, state: Dict[str, Any]) -> FactCheckReport:
        """Generate comprehensive fact-check report.

        Args:
            state: Complete state from multi-agent workflow

        Returns:
            FactCheckReport with all analysis results
        """
        self.logger.info(
            "generating_report",
            claim_id=state.get("claim_id", "unknown"),
            verdict=state.get("final_verdict"),
        )

        # Extract key information
        claim_id = state.get("claim_id", f"claim_{datetime.now().timestamp()}")
        original_claim = state.get("original_claim", "")
        classification = state.get("classification", {})
        evidence = state.get("evidence_retrieved", [])
        assertion_verdicts = state.get("triplet_verdicts", {})
        anomaly_scores = state.get("anomaly_scores", {})

        # Calculate confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(
            assertion_verdicts, anomaly_scores, evidence
        )

        # Determine verdict
        verdict = self._determine_final_verdict(state)
        overall_confidence = state.get("confidence", confidence_breakdown["overall"])

        # Extract sources
        sources_used = list(set([e.get("source", "unknown") for e in evidence]))
        high_credibility_sources = [
            s for s in sources_used if self._get_source_credibility(s) > 0.8
        ]

        # Generate key findings
        key_findings = self._generate_key_findings(
            verdict, assertion_verdicts, evidence
        )

        # Identify contradictions
        contradictions = self._identify_contradictions(assertion_verdicts, evidence)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            verdict, overall_confidence, classification
        )

        # Create report
        report = FactCheckReport(
            claim_id=claim_id,
            original_claim=original_claim,
            verdict=verdict,
            confidence=overall_confidence,
            theme=classification.get("theme", "unknown"),
            complexity=classification.get("complexity", 0),
            urgency=classification.get("urgency", 0),
            evidence_count=len(evidence),
            evidence_summary=self._summarize_evidence(evidence),
            sources_used=sources_used,
            high_credibility_sources=high_credibility_sources,
            decomposed_assertions=state.get("decomposed_assertions", []),
            assertion_verdicts=assertion_verdicts,
            anomaly_scores=anomaly_scores,
            reasoning_trace=state.get("reasoning_trace", []),
            key_findings=key_findings,
            contradictions=contradictions,
            agents_involved=state.get("agents_involved", []),
            processing_time_seconds=state.get("processing_time", 0.0),
            created_at=datetime.fromisoformat(state.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.now(),
            recommended_actions=recommendations,
            confidence_breakdown=confidence_breakdown,
        )

        self.logger.info(
            "report_generated",
            claim_id=claim_id,
            verdict=verdict.value,
            confidence=overall_confidence,
            evidence_count=len(evidence),
        )

        return report

    def _determine_final_verdict(self, state: Dict[str, Any]) -> VerdictType:
        """Determine final verdict from state."""
        verdict_str = state.get("final_verdict", "INSUFFICIENT_INFO")

        try:
            return VerdictType(verdict_str)
        except ValueError:
            self.logger.warning(
                "unknown_verdict", verdict=verdict_str, defaulting_to="INSUFFICIENT_INFO"
            )
            return VerdictType.INSUFFICIENT_INFO

    def _calculate_confidence_breakdown(
        self,
        assertion_verdicts: Dict[str, Any],
        anomaly_scores: Dict[str, float],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate detailed confidence breakdown."""
        if not assertion_verdicts:
            return {"overall": 0.5, "assertion_confidence": 0.5, "evidence_quality": 0.5}

        # Assertion confidence
        confidences = []
        for verdict_data in assertion_verdicts.values():
            if isinstance(verdict_data, dict):
                confidences.append(verdict_data.get("confidence", 0.5))
            else:
                confidences.append(0.5)

        assertion_conf = sum(confidences) / len(confidences) if confidences else 0.5

        # Evidence quality
        if evidence:
            credibility_scores = [
                e.get("credibility", 0.5) for e in evidence
            ]
            evidence_quality = sum(credibility_scores) / len(credibility_scores)
        else:
            evidence_quality = 0.0

        # Anomaly penalty
        avg_anomaly = (
            sum(anomaly_scores.values()) / len(anomaly_scores)
            if anomaly_scores
            else 0.0
        )

        # Overall confidence
        overall = (assertion_conf * 0.6 + evidence_quality * 0.4) * (1 - avg_anomaly * 0.3)

        return {
            "overall": round(overall, 3),
            "assertion_confidence": round(assertion_conf, 3),
            "evidence_quality": round(evidence_quality, 3),
            "anomaly_penalty": round(avg_anomaly, 3),
        }

    def _get_source_credibility(self, source: str) -> float:
        """Get credibility score for a source."""
        from config.settings import settings

        return settings.SOURCE_CREDIBILITY_WEIGHTS.get(source, 0.5)

    def _summarize_evidence(
        self, evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create concise evidence summary."""
        summary = []
        for e in evidence[:10]:  # Limit to top 10
            summary.append(
                {
                    "source": e.get("source", "unknown"),
                    "credibility": e.get("credibility", 0.5),
                    "text_preview": e.get("text", "")[:200] + "...",
                }
            )
        return summary

    def _generate_key_findings(
        self,
        verdict: VerdictType,
        assertion_verdicts: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate key findings from analysis."""
        findings = []

        # Verdict summary
        findings.append(f"Overall claim is {verdict.value}")

        # Assertion analysis
        if assertion_verdicts:
            supported_count = sum(
                1
                for v in assertion_verdicts.values()
                if isinstance(v, dict) and v.get("verdict") == "SUPPORTED"
            )
            refuted_count = sum(
                1
                for v in assertion_verdicts.values()
                if isinstance(v, dict) and v.get("verdict") == "REFUTED"
            )

            if supported_count > 0:
                findings.append(
                    f"{supported_count}/{len(assertion_verdicts)} assertions are supported"
                )
            if refuted_count > 0:
                findings.append(
                    f"{refuted_count}/{len(assertion_verdicts)} assertions are refuted"
                )

        # Evidence quality
        high_quality_count = sum(1 for e in evidence if e.get("credibility", 0) > 0.8)
        if high_quality_count > 0:
            findings.append(
                f"{high_quality_count} high-credibility sources found"
            )

        return findings

    def _identify_contradictions(
        self, assertion_verdicts: Dict[str, Any], evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify contradictions in evidence or verdicts."""
        contradictions = []

        # Check for conflicting verdicts
        verdicts = [
            v.get("verdict") if isinstance(v, dict) else None
            for v in assertion_verdicts.values()
        ]
        if "SUPPORTED" in verdicts and "REFUTED" in verdicts:
            contradictions.append(
                "Conflicting verdicts found across different assertions"
            )

        # Could add more sophisticated contradiction detection here

        return contradictions

    def _generate_recommendations(
        self, verdict: VerdictType, confidence: float, classification: Dict[str, Any]
    ) -> List[str]:
        """Generate recommended actions based on results."""
        recommendations = []

        if confidence < 0.6:
            recommendations.append(
                "LOW CONFIDENCE: Consider manual review by fact-checking expert"
            )

        if verdict == VerdictType.REFUTED and confidence > 0.8:
            recommendations.append(
                "HIGH PRIORITY: Publish correction with supporting evidence"
            )

        if verdict == VerdictType.CONFLICTING:
            recommendations.append(
                "CONFLICTING EVIDENCE: Conduct deeper investigation with additional sources"
            )

        if classification.get("urgency", 0) > 7:
            recommendations.append(
                "URGENT: Fast-track public communication due to high urgency score"
            )

        if verdict == VerdictType.REQUIRES_HUMAN_REVIEW:
            recommendations.append(
                "ESCALATE: Requires human expert review before publication"
            )

        return recommendations

    def generate_alert(
        self,
        report: FactCheckReport,
        stakeholders: Optional[List[str]] = None,
    ) -> Optional[Alert]:
        """Generate alert if necessary based on report.

        Args:
            report: Fact-check report
            stakeholders: List of stakeholder types to notify

        Returns:
            Alert object if alerting is needed, None otherwise
        """
        # Determine alert level
        alert_level = self._determine_alert_level(report)

        if alert_level is None:
            return None

        # Default stakeholders
        if stakeholders is None:
            stakeholders = self._determine_stakeholders(report)

        # Create alert message
        message = self._create_alert_message(report, alert_level)

        alert = Alert(
            level=alert_level,
            message=message,
            claim_id=report.claim_id,
            verdict=report.verdict,
            confidence=report.confidence,
            reasoning="\n".join(report.key_findings),
            created_at=datetime.now(),
            stakeholders=stakeholders,
            metadata={
                "urgency": report.urgency,
                "complexity": report.complexity,
                "evidence_count": report.evidence_count,
            },
        )

        self.logger.warning(
            "alert_generated",
            claim_id=report.claim_id,
            level=alert_level.value,
            stakeholders=stakeholders,
        )

        return alert

    def _determine_alert_level(self, report: FactCheckReport) -> Optional[AlertLevel]:
        """Determine if alert is needed and at what level."""
        # Critical: High-confidence refutation
        if report.verdict == VerdictType.REFUTED and report.confidence > 0.85:
            return AlertLevel.CRITICAL

        # Urgent: High urgency and high confidence
        if report.urgency > 8 and report.confidence > 0.7:
            return AlertLevel.URGENT

        # Warning: Medium confidence issues
        if report.verdict in [VerdictType.REFUTED, VerdictType.CONFLICTING]:
            if report.confidence > 0.6:
                return AlertLevel.WARNING

        # Info: Low-confidence or insufficient info
        if report.verdict == VerdictType.INSUFFICIENT_INFO:
            return AlertLevel.INFO

        return None

    def _determine_stakeholders(self, report: FactCheckReport) -> List[str]:
        """Determine who should be notified."""
        stakeholders = []

        if report.verdict == VerdictType.REFUTED and report.confidence > 0.8:
            stakeholders.extend(["journalists", "public"])

        if report.urgency > 7:
            stakeholders.append("regulators")

        if report.theme in ["health", "politics", "climate"]:
            stakeholders.append("domain_experts")

        return stakeholders or ["internal"]

    def _create_alert_message(
        self, report: FactCheckReport, level: AlertLevel
    ) -> str:
        """Create human-readable alert message."""
        templates = {
            AlertLevel.CRITICAL: f"🚨 CRITICAL: Claim REFUTED with {report.confidence:.0%} confidence",
            AlertLevel.URGENT: f"⚠️ URGENT: {report.verdict.value} - Immediate attention required",
            AlertLevel.WARNING: f"⚡ WARNING: {report.verdict.value} detected",
            AlertLevel.INFO: f"ℹ️ INFO: Fact-check completed - {report.verdict.value}",
        }

        base_message = templates.get(level, f"Fact-check result: {report.verdict.value}")

        details = (
            f"\n\nClaim: {report.original_claim[:150]}..."
            f"\nTheme: {report.theme}"
            f"\nConfidence: {report.confidence:.0%}"
            f"\nEvidence sources: {len(report.sources_used)}"
        )

        return base_message + details
