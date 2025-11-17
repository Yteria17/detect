"""
Agent 4 : Fact-Checker (Vérificateur de Faits)

Responsabilités:
- Vérification multimodale des affirmations
- Cross-referencing intelligent
- Chain-of-Thought reasoning
- Graph-based verification pour claims complexes
- Résolution de preuves contradictoires
"""

from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from collections import Counter


class FactCheckerAgent:
    """
    Agent spécialisé dans la vérification factuelle avancée.
    Utilise Chain-of-Thought reasoning et Graph-based verification.
    """

    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize the Fact Checker Agent.

        Args:
            llm_client: Client LLM pour reasoning
            config: Configuration de l'agent
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

    def verify(self, state: Dict) -> Dict:
        """
        Vérification principale des assertions.

        Args:
            state: État partagé avec assertions et evidence

        Returns:
            État mis à jour avec verdicts
        """
        assertions = state.get('decomposed_assertions', [])
        evidence_pool = state.get('evidence_retrieved', [])
        classification = state.get('classification', {})

        complexity = classification.get('complexity', 5)

        # Choix de la stratégie selon la complexité
        if complexity > 7:
            # Utiliser Graph-based verification pour claims complexes
            verdicts = self._graph_based_verification(assertions, evidence_pool, state)
        else:
            # Utiliser Chain-of-Thought standard
            verdicts = self._chain_of_thought_verification(assertions, evidence_pool)

        state['assertion_verdicts'] = verdicts

        # Calcul du verdict final
        final_verdict = self._compute_final_verdict(verdicts)
        state.update(final_verdict)

        # Trace
        if 'reasoning_trace' not in state:
            state['reasoning_trace'] = []

        state['reasoning_trace'].append(
            f"FactChecker: Verified {len(verdicts)} assertions. "
            f"Final verdict: {final_verdict['final_verdict']}"
        )

        return state

    def _chain_of_thought_verification(
        self,
        assertions: List[str],
        evidence_pool: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Vérification via Chain-of-Thought reasoning.

        Args:
            assertions: Liste d'assertions à vérifier
            evidence_pool: Pool de preuves disponibles

        Returns:
            Dict mapping assertion -> verdict
        """
        verdicts = {}

        for assertion in assertions:
            # Filtrer evidence pertinente pour cette assertion
            relevant_evidence = [
                e for e in evidence_pool
                if e.get('assertion') == assertion
            ]

            # Vérification avec CoT
            verdict = self._verify_single_assertion_cot(
                assertion,
                relevant_evidence
            )

            verdicts[assertion] = verdict

        return verdicts

    def _verify_single_assertion_cot(
        self,
        assertion: str,
        evidence: List[Dict]
    ) -> Dict:
        """
        Vérifie une assertion unique avec Chain-of-Thought.

        Args:
            assertion: Assertion à vérifier
            evidence: Preuves pertinentes

        Returns:
            Verdict détaillé
        """
        if not evidence:
            return {
                'verdict': 'INSUFFICIENT_INFO',
                'confidence': 0.0,
                'reasoning': 'Aucune preuve disponible',
                'supporting_evidence': [],
                'refuting_evidence': []
            }

        # Tri des preuves par qualité
        sorted_evidence = sorted(
            evidence,
            key=lambda e: e.get('credibility', 0.5) * e.get('relevance_score', 0.5),
            reverse=True
        )

        # Utiliser les meilleures preuves
        top_evidence = sorted_evidence[:5]

        if self.llm_client:
            return self._llm_cot_verification(assertion, top_evidence)
        else:
            return self._heuristic_verification(assertion, top_evidence)

    def _llm_cot_verification(
        self,
        assertion: str,
        evidence: List[Dict]
    ) -> Dict:
        """
        Vérification LLM avec Chain-of-Thought.

        Args:
            assertion: Assertion
            evidence: Preuves

        Returns:
            Verdict
        """
        evidence_text = "\n\n".join([
            f"Source {i+1} ({e['source']}, crédibilité: {e['credibility']:.2f}):\n{e['text']}"
            for i, e in enumerate(evidence)
        ])

        prompt = f"""Vous êtes un fact-checker expert. Vérifiez cette affirmation étape par étape.

Affirmation à vérifier: {assertion}

Preuves disponibles:
{evidence_text}

Étapes de vérification:
1. Identifiez les assertions factuelles clés
2. Pour chaque preuve, évaluez sa pertinence et crédibilité
3. Déterminez si les preuves SUPPORTENT, RÉFUTENT ou sont INSUFFISANTES
4. Résolvez les contradictions éventuelles en pondérant par crédibilité
5. Conclusion finale avec niveau de confiance

Répondez au format JSON:
{{
    "key_facts": ["fait 1", "fait 2", ...],
    "evidence_analysis": [
        {{"source": "...", "supports": true/false, "weight": 0-1}}
    ],
    "reasoning_steps": ["étape 1", "étape 2", ...],
    "verdict": "SUPPORTED" | "REFUTED" | "INSUFFICIENT_INFO",
    "confidence": <float 0-1>
}}
"""

        try:
            if hasattr(self.llm_client, 'invoke'):
                response = self.llm_client.invoke(prompt)
            else:
                return self._heuristic_verification(assertion, evidence)

            if isinstance(response, str):
                result = json.loads(response)
            else:
                result = response

            # Formatage du résultat
            return {
                'verdict': result.get('verdict', 'INSUFFICIENT_INFO'),
                'confidence': result.get('confidence', 0.5),
                'reasoning': '\n'.join(result.get('reasoning_steps', [])),
                'key_facts': result.get('key_facts', []),
                'supporting_evidence': [
                    e for e in evidence
                    if any(
                        ea.get('source') == e['source'] and ea.get('supports')
                        for ea in result.get('evidence_analysis', [])
                    )
                ],
                'refuting_evidence': [
                    e for e in evidence
                    if any(
                        ea.get('source') == e['source'] and not ea.get('supports')
                        for ea in result.get('evidence_analysis', [])
                    )
                ]
            }

        except Exception as e:
            print(f"Erreur LLM CoT: {e}")
            return self._heuristic_verification(assertion, evidence)

    def _heuristic_verification(
        self,
        assertion: str,
        evidence: List[Dict]
    ) -> Dict:
        """
        Vérification heuristique simple (fallback sans LLM).

        Args:
            assertion: Assertion
            evidence: Preuves

        Returns:
            Verdict
        """
        # Calculer un score basé sur crédibilité moyenne
        avg_credibility = sum(e['credibility'] for e in evidence) / len(evidence)

        # Score de pertinence
        avg_relevance = sum(e.get('relevance_score', 0.5) for e in evidence) / len(evidence)

        # Score combiné
        combined_score = (avg_credibility * 0.6 + avg_relevance * 0.4)

        # Détermination du verdict
        if combined_score > 0.7 and len(evidence) >= 2:
            verdict = 'SUPPORTED'
            confidence = combined_score
        elif combined_score < 0.3:
            verdict = 'REFUTED'
            confidence = 1.0 - combined_score
        else:
            verdict = 'INSUFFICIENT_INFO'
            confidence = 0.5

        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': f'Heuristique: {len(evidence)} preuves, score moyen {combined_score:.2f}',
            'supporting_evidence': evidence if verdict == 'SUPPORTED' else [],
            'refuting_evidence': evidence if verdict == 'REFUTED' else []
        }

    def _graph_based_verification(
        self,
        assertions: List[str],
        evidence_pool: List[Dict],
        state: Dict
    ) -> Dict[str, Dict]:
        """
        Vérification basée sur graphes pour claims complexes.

        Args:
            assertions: Assertions à vérifier
            evidence_pool: Pool de preuves
            state: État global

        Returns:
            Verdicts
        """
        # Pour claims complexes, on crée un graphe de dépendances
        # entre assertions et on vérifie dans l'ordre topologique

        # Version simplifiée: vérification séquentielle avec contexte cumulatif
        verdicts = {}
        verified_facts = []

        for i, assertion in enumerate(assertions):
            relevant_evidence = [
                e for e in evidence_pool
                if e.get('assertion') == assertion
            ]

            # Ajouter le contexte des faits déjà vérifiés
            context = {
                'verified_facts': verified_facts,
                'previous_verdicts': verdicts
            }

            verdict = self._verify_with_context(
                assertion,
                relevant_evidence,
                context
            )

            verdicts[assertion] = verdict

            # Si vérifié positivement, ajouter aux faits connus
            if verdict['verdict'] == 'SUPPORTED' and verdict['confidence'] > 0.7:
                verified_facts.append(assertion)

        return verdicts

    def _verify_with_context(
        self,
        assertion: str,
        evidence: List[Dict],
        context: Dict
    ) -> Dict:
        """
        Vérifie avec contexte des vérifications précédentes.

        Args:
            assertion: Assertion
            evidence: Preuves
            context: Contexte (faits déjà vérifiés)

        Returns:
            Verdict
        """
        # Vérification standard
        base_verdict = self._verify_single_assertion_cot(assertion, evidence)

        # Ajustement selon le contexte
        verified_facts = context.get('verified_facts', [])

        # Si cette assertion contredit des faits vérifiés, baisser confiance
        if self._contradicts_verified_facts(assertion, verified_facts):
            base_verdict['confidence'] *= 0.7
            base_verdict['reasoning'] += '\nATTENTION: Contradiction potentielle avec faits vérifiés'

        return base_verdict

    def _contradicts_verified_facts(
        self,
        assertion: str,
        verified_facts: List[str]
    ) -> bool:
        """
        Détecte si assertion contredit des faits déjà vérifiés.

        Args:
            assertion: Assertion à tester
            verified_facts: Faits déjà vérifiés

        Returns:
            True si contradiction
        """
        # Détection simple de négation
        negation_words = ['ne pas', 'n\'est pas', 'jamais', 'aucun', 'faux']

        for fact in verified_facts:
            # Si l'assertion contient le fait mais avec négation
            if any(neg in assertion.lower() for neg in negation_words):
                if any(word in assertion.lower() for word in fact.lower().split()):
                    return True

        return False

    def _compute_final_verdict(self, assertion_verdicts: Dict[str, Dict]) -> Dict:
        """
        Calcule le verdict final à partir des verdicts individuels.

        Args:
            assertion_verdicts: Verdicts de chaque assertion

        Returns:
            Verdict final global
        """
        if not assertion_verdicts:
            return {
                'final_verdict': 'INSUFFICIENT_INFO',
                'confidence': 0.0,
                'explanation': 'Aucune assertion à vérifier'
            }

        # Comptage des verdicts
        verdict_counts = Counter(
            v['verdict'] for v in assertion_verdicts.values()
        )

        total = len(assertion_verdicts)
        refuted = verdict_counts.get('REFUTED', 0)
        supported = verdict_counts.get('SUPPORTED', 0)
        insufficient = verdict_counts.get('INSUFFICIENT_INFO', 0)

        # Logique de décision:
        # - Si au moins 1 assertion REFUTED avec haute confiance -> REFUTED
        # - Si toutes SUPPORTED -> SUPPORTED
        # - Sinon -> INSUFFICIENT_INFO ou PARTIALLY_SUPPORTED

        high_confidence_refuted = any(
            v['verdict'] == 'REFUTED' and v['confidence'] > 0.7
            for v in assertion_verdicts.values()
        )

        if high_confidence_refuted:
            final = 'REFUTED'
            confidence = max(
                v['confidence']
                for v in assertion_verdicts.values()
                if v['verdict'] == 'REFUTED'
            )
            explanation = f'{refuted}/{total} assertions réfutées'

        elif supported == total and all(
            v['confidence'] > 0.6 for v in assertion_verdicts.values()
        ):
            final = 'SUPPORTED'
            confidence = sum(
                v['confidence'] for v in assertion_verdicts.values()
            ) / total
            explanation = f'{supported}/{total} assertions supportées'

        elif supported > refuted and supported > insufficient:
            final = 'PARTIALLY_SUPPORTED'
            confidence = supported / total
            explanation = f'{supported}/{total} assertions supportées, {refuted} réfutées'

        else:
            final = 'INSUFFICIENT_INFO'
            confidence = 0.5
            explanation = f'Preuves insuffisantes ({insufficient}/{total} assertions)'

        return {
            'final_verdict': final,
            'confidence': confidence,
            'explanation': explanation,
            'verdict_breakdown': {
                'supported': supported,
                'refuted': refuted,
                'insufficient': insufficient,
                'total': total
            }
        }

    def resolve_conflicting_evidence(
        self,
        evidence_supporting: List[Dict],
        evidence_refuting: List[Dict]
    ) -> Dict:
        """
        Résout les preuves contradictoires en pondérant par crédibilité.

        Args:
            evidence_supporting: Preuves qui supportent
            evidence_refuting: Preuves qui réfutent

        Returns:
            Résolution avec verdict pondéré
        """
        support_score = sum(e['credibility'] for e in evidence_supporting)
        refute_score = sum(e['credibility'] for e in evidence_refuting)

        total = support_score + refute_score

        if total == 0:
            return {
                'verdict': 'INSUFFICIENT_INFO',
                'confidence': 0.0,
                'conflicting_evidence_detected': True
            }

        if support_score > refute_score:
            return {
                'verdict': 'SUPPORTED',
                'confidence': support_score / total,
                'conflicting_evidence_detected': True,
                'reasoning': f'Score support: {support_score:.2f} vs réfutation: {refute_score:.2f}'
            }
        else:
            return {
                'verdict': 'REFUTED',
                'confidence': refute_score / total,
                'conflicting_evidence_detected': True,
                'reasoning': f'Score réfutation: {refute_score:.2f} vs support: {support_score:.2f}'
            }
