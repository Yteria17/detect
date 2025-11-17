"""Agent 5: Reporter - Consolidates results and generates reports."""

from typing import Dict, List
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.types import FactCheckingState, Verdict
from utils.logger import log
from utils.helpers import format_reasoning_trace, calculate_confidence_score


class ReporterAgent(BaseAgent):
    """
    Reporter Agent.

    Responsibilities:
    - Consolidate verdicts from all agents
    - Generate final verdict with confidence score
    - Create structured reports
    - Determine alert levels and escalation needs
    - Maintain audit trail
    """

    def __init__(self):
        super().__init__(
            name="ReporterAgent",
            description="Consolidates results and generates comprehensive reports"
        )

    async def process(self, state: FactCheckingState) -> FactCheckingState:
        """
        Generate final report and verdict.

        Args:
            state: Current fact-checking state

        Returns:
            Updated state with final verdict and report
        """
        self.log_action(state, "Starting report generation")

        # Step 1: Consolidate assertion verdicts
        final_verdict, confidence = self._consolidate_verdicts(state)

        state.final_verdict = final_verdict
        state.confidence = confidence

        self.log_action(
            state,
            f"Final verdict: {final_verdict.value} (confidence: {confidence:.2f})"
        )

        # Step 2: Determine if escalation is needed
        needs_escalation = self._check_escalation_needed(state)

        if needs_escalation:
            self.log_action(state, "⚠️ Escalation recommended: Low confidence or high complexity")

        # Step 3: Generate explanation
        explanation = self._generate_explanation(state)
        self.log_action(state, f"Generated detailed explanation ({len(explanation)} chars)")

        # Update timestamp
        state.updated_at = datetime.now()

        return state

    def _consolidate_verdicts(self, state: FactCheckingState) -> tuple[Verdict, float]:
        """
        Consolidate individual assertion verdicts into final verdict.

        Args:
            state: Current state

        Returns:
            Tuple of (final_verdict, confidence_score)
        """
        if not state.decomposed_assertions:
            return Verdict.INSUFFICIENT_INFO, 0.0

        # Count verdicts
        verdict_counts = {
            Verdict.SUPPORTED: 0,
            Verdict.REFUTED: 0,
            Verdict.INSUFFICIENT_INFO: 0,
            Verdict.CONFLICTING: 0
        }

        total_confidence = 0.0
        total_assertions = len(state.decomposed_assertions)

        for assertion in state.decomposed_assertions:
            verdict_counts[assertion.verdict] += 1
            total_confidence += assertion.confidence

        # Decision logic
        refuted_count = verdict_counts[Verdict.REFUTED]
        supported_count = verdict_counts[Verdict.SUPPORTED]
        insufficient_count = verdict_counts[Verdict.INSUFFICIENT_INFO]

        # If ANY assertion is refuted with confidence, claim is refuted
        if refuted_count > 0:
            # Check confidence of refuted assertions
            refuted_assertions = [a for a in state.decomposed_assertions if a.verdict == Verdict.REFUTED]
            avg_refuted_confidence = sum(a.confidence for a in refuted_assertions) / len(refuted_assertions)

            if avg_refuted_confidence > 0.6:
                return Verdict.REFUTED, avg_refuted_confidence
            else:
                return Verdict.CONFLICTING, 0.5

        # If ALL assertions are supported, claim is supported
        if supported_count == total_assertions:
            avg_confidence = total_confidence / total_assertions
            return Verdict.SUPPORTED, avg_confidence

        # If MOST assertions are supported (>70%)
        if supported_count / total_assertions > 0.7:
            avg_confidence = total_confidence / total_assertions
            return Verdict.SUPPORTED, avg_confidence * 0.85  # Reduce confidence slightly

        # If too many insufficient
        if insufficient_count / total_assertions > 0.5:
            return Verdict.INSUFFICIENT_INFO, 0.3

        # Mixed results
        return Verdict.CONFLICTING, total_confidence / total_assertions

    def _check_escalation_needed(self, state: FactCheckingState) -> bool:
        """
        Determine if manual review/escalation is needed.

        Args:
            state: Current state

        Returns:
            True if escalation needed
        """
        # Escalate if confidence is low
        if state.confidence < 0.7:
            return True

        # Escalate if high complexity
        if state.classification and state.classification.complexity > 8:
            return True

        # Escalate if conflicting verdict
        if state.final_verdict == Verdict.CONFLICTING:
            return True

        # Escalate if high urgency and not definitively resolved
        if (state.classification and
            state.classification.urgency > 7 and
            state.final_verdict == Verdict.INSUFFICIENT_INFO):
            return True

        return False

    def _generate_explanation(self, state: FactCheckingState) -> str:
        """
        Generate human-readable explanation of the verdict.

        Args:
            state: Current state

        Returns:
            Explanation text
        """
        lines = []

        # Header
        lines.append(f"# Fact-Check Report: {state.claim_id}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        # Original claim
        lines.append("## Original Claim")
        lines.append(f'"{state.original_claim}"')
        lines.append("")

        # Final verdict
        lines.append("## Final Verdict")
        lines.append(f"**{state.final_verdict.value}** (Confidence: {state.confidence:.0%})")
        lines.append("")

        # Classification
        if state.classification:
            lines.append("## Classification")
            lines.append(f"- Theme: {state.classification.theme}")
            lines.append(f"- Complexity: {state.classification.complexity}/10")
            lines.append(f"- Urgency: {state.classification.urgency}/10")
            if state.classification.detected_entities:
                lines.append(f"- Key Entities: {', '.join(state.classification.detected_entities[:5])}")
            lines.append("")

        # Assertion Analysis
        lines.append("## Assertion Analysis")
        for i, assertion in enumerate(state.decomposed_assertions, 1):
            lines.append(f"\n### Assertion {i}")
            lines.append(f'"{assertion.text}"')
            lines.append(f"- Verdict: **{assertion.verdict.value}**")
            lines.append(f"- Confidence: {assertion.confidence:.0%}")

            # Anomaly score if available
            if assertion.assertion_id in state.anomaly_scores:
                anomaly = state.anomaly_scores[assertion.assertion_id]
                lines.append(f"- Anomaly Score: {anomaly.score:.2f}")
                if anomaly.detected_patterns:
                    lines.append(f"- Detected Patterns: {', '.join(anomaly.detected_patterns)}")

            # Evidence count
            if assertion.evidence:
                lines.append(f"- Evidence Sources: {len(assertion.evidence)}")

        # Key Evidence
        if state.evidence_retrieved:
            lines.append("\n## Key Evidence Sources")
            # Get top 5 highest credibility sources
            top_sources = sorted(
                state.evidence_retrieved,
                key=lambda e: e.source.credibility_score,
                reverse=True
            )[:5]

            for i, evidence in enumerate(top_sources, 1):
                lines.append(f"\n{i}. **{evidence.source.domain}** (credibility: {evidence.source.credibility_score:.0%})")
                if evidence.source.title:
                    lines.append(f"   {evidence.source.title}")
                lines.append(f"   {evidence.source.url}")

        # Reasoning trace
        lines.append("\n## Processing Trace")
        lines.append(format_reasoning_trace(state.reasoning_trace))

        # Recommendation
        lines.append("\n## Recommendation")
        if state.final_verdict == Verdict.SUPPORTED:
            lines.append("✅ This claim appears to be supported by available evidence.")
        elif state.final_verdict == Verdict.REFUTED:
            lines.append("❌ This claim appears to be false or misleading based on evidence.")
        elif state.final_verdict == Verdict.CONFLICTING:
            lines.append("⚠️ Evidence is conflicting. Further investigation recommended.")
        else:
            lines.append("❓ Insufficient evidence to make a determination.")

        if self._check_escalation_needed(state):
            lines.append("\n⚠️ **Manual review recommended** due to low confidence or high complexity.")

        return "\n".join(lines)

    def generate_json_report(self, state: FactCheckingState) -> Dict:
        """
        Generate structured JSON report.

        Args:
            state: Final state

        Returns:
            Report dictionary
        """
        return {
            'claim_id': state.claim_id,
            'claim': state.original_claim,
            'verdict': state.final_verdict.value,
            'confidence': state.confidence,
            'classification': {
                'theme': state.classification.theme if state.classification else 'unknown',
                'complexity': state.classification.complexity if state.classification else 0,
                'urgency': state.classification.urgency if state.classification else 0,
            } if state.classification else None,
            'assertions': [
                {
                    'text': a.text,
                    'verdict': a.verdict.value,
                    'confidence': a.confidence,
                    'evidence_count': len(a.evidence)
                }
                for a in state.decomposed_assertions
            ],
            'evidence_summary': {
                'total_sources': len(state.evidence_retrieved),
                'high_credibility_sources': len([
                    e for e in state.evidence_retrieved
                    if e.source.credibility_score > 0.75
                ]),
                'unique_domains': len(set(e.source.domain for e in state.evidence_retrieved))
            },
            'anomaly_detection': {
                'high_anomaly_count': len([
                    s for s in state.anomaly_scores.values()
                    if s.score > 0.6
                ]),
                'average_anomaly_score': sum(
                    s.score for s in state.anomaly_scores.values()
                ) / len(state.anomaly_scores) if state.anomaly_scores else 0.0
            } if state.anomaly_scores else None,
            'processing': {
                'agents_involved': state.agents_involved,
                'created_at': state.created_at.isoformat(),
                'updated_at': state.updated_at.isoformat() if state.updated_at else None,
                'processing_time_ms': state.processing_time_ms
            },
            'needs_escalation': self._check_escalation_needed(state),
            'reasoning_trace': state.reasoning_trace
        }
