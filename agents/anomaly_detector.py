"""
Agent 3 : Détecteur d'Anomalies Sémantiques

Responsabilités:
- Analyse de cohérence logique via LLM
- Détection de contradictions
- Analyse linguistique pour patterns manipulateurs
- Scoring de suspicion
"""

from typing import Dict, List, Optional
import json
from datetime import datetime


class AnomalyDetectorAgent:
    """
    Agent spécialisé dans la détection d'anomalies sémantiques et de patterns
    suspects dans les affirmations.
    """

    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize the Anomaly Detector Agent.

        Args:
            llm_client: Client LLM pour le reasoning (Claude, GPT-4, etc.)
            config: Configuration de l'agent
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.threshold_suspicious = self.config.get('threshold_suspicious', 0.7)
        self.detection_patterns = self._load_detection_patterns()

    def _load_detection_patterns(self) -> Dict:
        """
        Charge les patterns de détection de désinformation.

        Returns:
            Dict contenant les patterns suspects
        """
        return {
            'manipulation_indicators': [
                'certitude absolue sans nuance',
                'appel à la peur excessive',
                'urgence artificielle',
                'langage émotionnel extrême',
                'théories du complot',
                'censure prétendue',
                'autorité non vérifiable'
            ],
            'linguistic_red_flags': [
                'TOUS savent que',
                'PERSONNE ne peut nier',
                'ILS ne veulent pas que vous sachiez',
                'La VÉRITÉ cachée',
                '100% prouvé',
                'URGENT: PARTAGEZ',
                'Supprimé par les médias'
            ],
            'structural_anomalies': [
                'absence de sources vérifiables',
                'citations décontextualisées',
                'statistiques sans référence',
                'témoignages anonymes',
                'images manipulées'
            ]
        }

    def analyze(self, state: Dict) -> Dict:
        """
        Analyse principale pour détecter les anomalies sémantiques.

        Args:
            state: État partagé contenant les assertions décomposées

        Returns:
            État mis à jour avec anomaly_scores
        """
        assertions = state.get('decomposed_assertions', [])
        original_claim = state.get('original_claim', '')

        anomaly_scores = {}
        detailed_analysis = []

        for assertion in assertions:
            score, details = self._analyze_single_assertion(assertion, original_claim)
            anomaly_scores[assertion] = score
            detailed_analysis.append({
                'assertion': assertion,
                'anomaly_score': score,
                'details': details,
                'is_suspicious': score > self.threshold_suspicious
            })

        # Calcul du score global
        avg_score = sum(anomaly_scores.values()) / len(anomaly_scores) if anomaly_scores else 0.0

        state['anomaly_scores'] = anomaly_scores
        state['anomaly_analysis'] = {
            'detailed_results': detailed_analysis,
            'average_anomaly_score': avg_score,
            'high_risk_assertions': [
                a for a in detailed_analysis
                if a['anomaly_score'] > self.threshold_suspicious
            ],
            'timestamp': datetime.now().isoformat()
        }

        # Ajout au trace de raisonnement
        if 'reasoning_trace' not in state:
            state['reasoning_trace'] = []

        state['reasoning_trace'].append(
            f"AnomalyDetector: Analyzed {len(assertions)} assertions. "
            f"Average anomaly score: {avg_score:.2f}. "
            f"High-risk assertions: {len(state['anomaly_analysis']['high_risk_assertions'])}"
        )

        return state

    def _analyze_single_assertion(self, assertion: str, context: str) -> tuple[float, Dict]:
        """
        Analyse une assertion unique pour détecter des anomalies.

        Args:
            assertion: L'assertion à analyser
            context: Contexte original de l'affirmation

        Returns:
            Tuple (score, details) où score est entre 0 et 1
        """
        details = {
            'manipulation_score': 0.0,
            'linguistic_flags': [],
            'coherence_score': 0.0,
            'reasoning': ''
        }

        # 1. Détection de patterns de manipulation
        manipulation_score = self._detect_manipulation_patterns(assertion)
        details['manipulation_score'] = manipulation_score

        # 2. Analyse linguistique
        linguistic_flags = self._detect_linguistic_flags(assertion)
        details['linguistic_flags'] = linguistic_flags

        # 3. Analyse de cohérence via LLM (si disponible)
        if self.llm_client:
            coherence_analysis = self._llm_coherence_check(assertion, context)
            details['coherence_score'] = coherence_analysis.get('score', 0.5)
            details['reasoning'] = coherence_analysis.get('reasoning', '')
        else:
            # Fallback: analyse basique
            details['coherence_score'] = 0.5
            details['reasoning'] = 'LLM non disponible - analyse basique uniquement'

        # Calcul du score final (pondéré)
        final_score = (
            manipulation_score * 0.3 +
            (len(linguistic_flags) / 10) * 0.3 +  # Normalisé sur 10 flags max
            details['coherence_score'] * 0.4
        )

        return min(final_score, 1.0), details

    def _detect_manipulation_patterns(self, text: str) -> float:
        """
        Détecte les patterns de manipulation dans le texte.

        Args:
            text: Texte à analyser

        Returns:
            Score de manipulation entre 0 et 1
        """
        text_lower = text.lower()
        indicators = self.detection_patterns['manipulation_indicators']

        matches = 0
        for indicator in indicators:
            if any(word in text_lower for word in indicator.split()):
                matches += 1

        # Score basé sur le nombre de matches
        score = min(matches / len(indicators), 1.0)
        return score

    def _detect_linguistic_flags(self, text: str) -> List[str]:
        """
        Détecte les drapeaux rouges linguistiques.

        Args:
            text: Texte à analyser

        Returns:
            Liste des flags détectés
        """
        flags = []
        text_upper = text.upper()

        for flag_pattern in self.detection_patterns['linguistic_red_flags']:
            if flag_pattern.upper() in text_upper:
                flags.append(flag_pattern)

        # Détection de patterns additionnels
        if text.count('!') > 3:
            flags.append('Exclamation excessive')

        if text.isupper() and len(text) > 20:
            flags.append('TEXTE TOUT EN MAJUSCULES')

        return flags

    def _llm_coherence_check(self, assertion: str, context: str) -> Dict:
        """
        Utilise un LLM pour vérifier la cohérence logique.

        Args:
            assertion: Assertion à vérifier
            context: Contexte original

        Returns:
            Dict avec score et reasoning
        """
        prompt = f"""Analysez cette affirmation pour détecter des anomalies sémantiques et des incohérences logiques.

Contexte original: {context}
Assertion à analyser: {assertion}

Évaluez les aspects suivants:
1. Cohérence logique interne
2. Contradictions potentielles
3. Affirmations extraordinaires nécessitant preuves extraordinaires
4. Ton manipulateur ou émotionnel excessif
5. Absence de nuances (tout noir ou tout blanc)

Répondez au format JSON:
{{
    "score": <float entre 0 et 1, où 1 = très suspect>,
    "reasoning": "<explication détaillée de votre analyse>",
    "red_flags": [<liste des drapeaux rouges détectés>],
    "confidence": <float entre 0 et 1>
}}
"""

        try:
            # Appel au LLM
            if hasattr(self.llm_client, 'invoke'):
                response = self.llm_client.invoke(prompt)
            else:
                # Fallback si pas de LLM
                return {
                    'score': 0.5,
                    'reasoning': 'LLM non configuré correctement',
                    'red_flags': [],
                    'confidence': 0.3
                }

            # Parse la réponse JSON
            if isinstance(response, str):
                result = json.loads(response)
            else:
                result = response

            return result

        except Exception as e:
            return {
                'score': 0.5,
                'reasoning': f'Erreur lors de l\'analyse LLM: {str(e)}',
                'red_flags': [],
                'confidence': 0.0
            }

    def escalate_if_needed(self, state: Dict) -> Dict:
        """
        Décide si une escalade est nécessaire basée sur les anomalies détectées.

        Args:
            state: État actuel

        Returns:
            État avec décision d'escalade
        """
        anomaly_analysis = state.get('anomaly_analysis', {})
        high_risk_count = len(anomaly_analysis.get('high_risk_assertions', []))
        avg_score = anomaly_analysis.get('average_anomaly_score', 0.0)

        should_escalate = (
            high_risk_count >= 2 or  # Au moins 2 assertions à haut risque
            avg_score > 0.8  # Score moyen très élevé
        )

        state['escalation_needed'] = should_escalate

        if should_escalate:
            state['escalation_reason'] = (
                f"Anomalies significatives détectées: "
                f"{high_risk_count} assertions à haut risque, "
                f"score moyen {avg_score:.2f}"
            )

        return state
