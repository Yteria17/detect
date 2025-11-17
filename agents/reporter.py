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
