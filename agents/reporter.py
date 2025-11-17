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
"""
Agent 5 : Reporter (Gestionnaire d'Alertes et Rapporteur)

Responsabilités:
- Consolidation des décisions de tous les agents
- Génération de rapports structurés
- Décision d'alertes urgentes
- Maintien de l'historique
- Recommandations de corrections
"""

from typing import Dict, List, Optional
from datetime import datetime
import json


class ReporterAgent:
    """
    Agent spécialisé dans la consolidation finale et le reporting.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Reporter Agent.

        Args:
            config: Configuration de l'agent
        """
        self.config = config or {}
        self.alert_threshold = self.config.get('alert_threshold', 0.8)

    def generate_report(self, state: Dict) -> Dict:
        """
        Génère le rapport final consolidé.

        Args:
            state: État complet de la vérification

        Returns:
            État mis à jour avec rapport final
        """
        # Collecte de toutes les informations
        original_claim = state.get('original_claim', '')
        classification = state.get('classification', {})
        anomaly_analysis = state.get('anomaly_analysis', {})
        evidence = state.get('evidence_retrieved', [])
        assertion_verdicts = state.get('assertion_verdicts', {})
        final_verdict = state.get('final_verdict', 'UNKNOWN')
        confidence = state.get('confidence', 0.0)
        reasoning_trace = state.get('reasoning_trace', [])

        # Construction du rapport structuré
        report = {
            'metadata': {
                'claim_id': state.get('claim_id', self._generate_claim_id()),
                'timestamp': datetime.now().isoformat(),
                'processing_time': self._calculate_processing_time(state),
                'agents_involved': self._list_agents_involved(reasoning_trace)
            },

            'claim': {
                'original': original_claim,
                'decomposed_assertions': state.get('decomposed_assertions', []),
                'theme': classification.get('theme', 'unknown'),
                'complexity': classification.get('complexity', 0),
                'urgency': classification.get('urgency', 0)
            },

            'analysis': {
                'anomaly_detection': {
                    'average_score': anomaly_analysis.get('average_anomaly_score', 0.0),
                    'high_risk_assertions': len(anomaly_analysis.get('high_risk_assertions', [])),
                    'suspicious_patterns': self._extract_suspicious_patterns(anomaly_analysis)
                },

                'evidence_summary': {
                    'total_evidence': len(evidence),
                    'high_credibility_sources': len([
                        e for e in evidence if e.get('credibility', 0) > 0.8
                    ]),
                    'average_credibility': self._calculate_avg_credibility(evidence),
                    'sources_consulted': list(set(e.get('source', '') for e in evidence))
                },

                'verification_results': {
                    'assertions_verified': len(assertion_verdicts),
                    'verdict_breakdown': self._get_verdict_breakdown(assertion_verdicts),
                    'detailed_verdicts': assertion_verdicts
                }
            },

            'verdict': {
                'final_verdict': final_verdict,
                'confidence': confidence,
                'explanation': state.get('explanation', ''),
                'verdict_label': self._get_verdict_label(final_verdict),
                'confidence_label': self._get_confidence_label(confidence)
            },

            'recommendations': self._generate_recommendations(state),

            'alert': self._generate_alert_decision(state),

            'traceability': {
                'reasoning_trace': reasoning_trace,
                'full_state': self._sanitize_state_for_export(state)
            }
        }

        state['final_report'] = report

        # Trace finale
        if 'reasoning_trace' not in state:
            state['reasoning_trace'] = []

        state['reasoning_trace'].append(
            f"Reporter: Report generated. Verdict: {final_verdict} ({confidence:.2%} confidence)"
        )

        return state

    def _generate_claim_id(self) -> str:
        """Génère un ID unique pour la claim."""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _calculate_processing_time(self, state: Dict) -> str:
        """Calcule le temps de traitement."""
        created_at = state.get('created_at')
        if created_at:
            try:
                start = datetime.fromisoformat(created_at)
                duration = datetime.now() - start
                return f"{duration.total_seconds():.2f}s"
            except:
                pass
        return "N/A"

    def _list_agents_involved(self, reasoning_trace: List[str]) -> List[str]:
        """Liste les agents impliqués dans le traitement."""
        agents = set()
        for trace in reasoning_trace:
            if ':' in trace:
                agent_name = trace.split(':')[0].strip()
                agents.add(agent_name)
        return sorted(list(agents))

    def _extract_suspicious_patterns(self, anomaly_analysis: Dict) -> List[str]:
        """Extrait les patterns suspects détectés."""
        patterns = []
        high_risk = anomaly_analysis.get('high_risk_assertions', [])

        for assertion_data in high_risk:
            details = assertion_data.get('details', {})
            flags = details.get('linguistic_flags', [])
            patterns.extend(flags)

        return list(set(patterns))

    def _calculate_avg_credibility(self, evidence: List[Dict]) -> float:
        """Calcule la crédibilité moyenne des sources."""
        if not evidence:
            return 0.0
        return sum(e.get('credibility', 0.0) for e in evidence) / len(evidence)

    def _get_verdict_breakdown(self, assertion_verdicts: Dict) -> Dict:
        """Obtient la répartition des verdicts."""
        from collections import Counter
        verdicts = [v['verdict'] for v in assertion_verdicts.values()]
        counts = Counter(verdicts)

        return {
            'SUPPORTED': counts.get('SUPPORTED', 0),
            'REFUTED': counts.get('REFUTED', 0),
            'INSUFFICIENT_INFO': counts.get('INSUFFICIENT_INFO', 0),
            'PARTIALLY_SUPPORTED': counts.get('PARTIALLY_SUPPORTED', 0)
        }

    def _get_verdict_label(self, verdict: str) -> str:
        """Traduit le verdict en label lisible."""
        labels = {
            'SUPPORTED': '✓ Vérifié',
            'REFUTED': '✗ Faux',
            'INSUFFICIENT_INFO': '? Informations insuffisantes',
            'PARTIALLY_SUPPORTED': '≈ Partiellement vérifié',
            'UNKNOWN': '? Inconnu'
        }
        return labels.get(verdict, verdict)

    def _get_confidence_label(self, confidence: float) -> str:
        """Traduit la confiance en label."""
        if confidence >= 0.9:
            return 'Très élevée'
        elif confidence >= 0.7:
            return 'Élevée'
        elif confidence >= 0.5:
            return 'Modérée'
        elif confidence >= 0.3:
            return 'Faible'
        else:
            return 'Très faible'

    def _generate_recommendations(self, state: Dict) -> List[str]:
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []

        final_verdict = state.get('final_verdict', '')
        confidence = state.get('confidence', 0.0)
        anomaly_analysis = state.get('anomaly_analysis', {})

        # Recommandation selon le verdict
        if final_verdict == 'REFUTED':
            recommendations.append(
                "🚨 Cette affirmation a été identifiée comme fausse. "
                "Ne pas partager ni amplifier."
            )
            recommendations.append(
                "Signaler la désinformation aux plateformes concernées."
            )

        elif final_verdict == 'PARTIALLY_SUPPORTED':
            recommendations.append(
                "⚠️ Cette affirmation contient des éléments vrais et faux. "
                "Vérifier le contexte avant de partager."
            )

        elif final_verdict == 'INSUFFICIENT_INFO':
            recommendations.append(
                "❓ Informations insuffisantes pour vérifier. "
                "Attendre des sources plus fiables avant de conclure."
            )

        # Recommandation selon la confiance
        if confidence < 0.5:
            recommendations.append(
                "⚠️ Confiance faible dans cette vérification. "
                "Recommandé de consulter des fact-checkers professionnels."
            )

        # Recommandation selon les anomalies
        high_risk_count = len(anomaly_analysis.get('high_risk_assertions', []))
        if high_risk_count > 0:
            recommendations.append(
                f"🔍 {high_risk_count} assertion(s) présentent des patterns suspects. "
                "Vigilance accrue recommandée."
            )

        # Recommandation générale
        recommendations.append(
            "💡 Toujours croiser les sources et consulter des médias de référence."
        )

        return recommendations

    def _generate_alert_decision(self, state: Dict) -> Dict:
        """Décide si une alerte doit être émise."""
        final_verdict = state.get('final_verdict', '')
        confidence = state.get('confidence', 0.0)
        classification = state.get('classification', {})
        urgency = classification.get('urgency', 0)

        should_alert = False
        alert_level = 'INFO'
        alert_reason = []

        # Critères d'alerte
        if final_verdict == 'REFUTED' and confidence > self.alert_threshold:
            should_alert = True
            alert_level = 'HIGH'
            alert_reason.append('Désinformation confirmée avec haute confiance')

        if urgency >= 8:
            should_alert = True
            if alert_level != 'HIGH':
                alert_level = 'MEDIUM'
            alert_reason.append('Contenu marqué comme urgent')

        theme = classification.get('theme', '')
        if theme in ['santé', 'sécurité'] and final_verdict == 'REFUTED':
            should_alert = True
            alert_level = 'CRITICAL'
            alert_reason.append('Désinformation dans domaine sensible (santé/sécurité)')

        return {
            'should_alert': should_alert,
            'alert_level': alert_level,
            'alert_reason': alert_reason,
            'alert_targets': self._get_alert_targets(alert_level) if should_alert else []
        }

    def _get_alert_targets(self, alert_level: str) -> List[str]:
        """Détermine les destinataires de l'alerte."""
        targets = []

        if alert_level in ['HIGH', 'CRITICAL']:
            targets.append('fact_checkers')
            targets.append('moderators')

        if alert_level == 'CRITICAL':
            targets.append('regulators')
            targets.append('public_health_authorities')

        targets.append('dashboard')

        return targets

    def _sanitize_state_for_export(self, state: Dict) -> Dict:
        """Nettoie l'état pour export (retirer données sensibles si besoin)."""
        # Pour l'instant, on retourne tout
        # Dans un système de production, on pourrait filtrer certaines clés
        safe_state = {
            k: v for k, v in state.items()
            if k not in ['final_report']  # Éviter récursion
        }
        return safe_state

    def export_report_json(self, state: Dict, filepath: str) -> None:
        """
        Exporte le rapport en JSON.

        Args:
            state: État avec rapport
            filepath: Chemin du fichier de sortie
        """
        report = state.get('final_report', {})

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def export_report_markdown(self, state: Dict, filepath: str) -> None:
        """
        Exporte le rapport en Markdown.

        Args:
            state: État avec rapport
            filepath: Chemin du fichier de sortie
        """
        report = state.get('final_report', {})

        md_content = self._generate_markdown(report)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

    def _generate_markdown(self, report: Dict) -> str:
        """Génère un rapport Markdown."""
        verdict = report.get('verdict', {})
        claim = report.get('claim', {})
        analysis = report.get('analysis', {})
        recommendations = report.get('recommendations', [])

        md = f"""# Rapport de Fact-Checking

## Affirmation Analysée

**Claim ID:** {report.get('metadata', {}).get('claim_id', 'N/A')}
**Date:** {report.get('metadata', {}).get('timestamp', 'N/A')}
**Thème:** {claim.get('theme', 'N/A')}

> {claim.get('original', '')}

## Verdict Final

**{verdict.get('verdict_label', 'N/A')}**

- **Confiance:** {verdict.get('confidence', 0):.1%} ({verdict.get('confidence_label', '')})
- **Explication:** {verdict.get('explanation', '')}

## Analyse Détaillée

### Décomposition
"""
        for i, assertion in enumerate(claim.get('decomposed_assertions', []), 1):
            md += f"{i}. {assertion}\n"

        md += f"""
### Preuves Consultées

- **Total:** {analysis.get('evidence_summary', {}).get('total_evidence', 0)}
- **Sources haute crédibilité:** {analysis.get('evidence_summary', {}).get('high_credibility_sources', 0)}
- **Crédibilité moyenne:** {analysis.get('evidence_summary', {}).get('average_credibility', 0):.2%}

### Détection d'Anomalies

- **Score moyen:** {analysis.get('anomaly_detection', {}).get('average_score', 0):.2f}
- **Assertions à risque:** {analysis.get('anomaly_detection', {}).get('high_risk_assertions', 0)}

## Recommandations

"""
        for rec in recommendations:
            md += f"- {rec}\n"

        md += "\n---\n\n*Rapport généré automatiquement par le système multi-agents de détection de désinformation*\n"

        return md
