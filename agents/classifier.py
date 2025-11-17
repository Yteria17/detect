"""
Agent 1 : Classificateur Thématique

Responsabilités:
- Décomposition des affirmations en assertions vérifiables
- Classification thématique
- Évaluation de la complexité
- Extraction d'entités nommées
"""

from typing import Dict, List, Optional
import json
from datetime import datetime


class ClassifierAgent:
    """
    Agent spécialisé dans la classification et la décomposition des affirmations.
    """

    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize the Classifier Agent.

        Args:
            llm_client: Client LLM pour le reasoning
            config: Configuration de l'agent
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.themes = [
            'politique', 'santé', 'climat', 'finance',
            'technologie', 'science', 'société', 'international'
        ]

    def classify(self, state: Dict) -> Dict:
        """
        Classification et décomposition principale.

        Args:
            state: État partagé contenant original_claim

        Returns:
            État mis à jour avec classification et assertions décomposées
        """
        claim = state.get('original_claim', '')

        if not claim:
            raise ValueError("original_claim manquant dans l'état")

        # Décomposition en assertions
        assertions = self._decompose_claim(claim)

        # Classification thématique
        theme = self._classify_theme(claim)

        # Évaluation de complexité
        complexity = self._evaluate_complexity(claim, assertions)

        # Évaluation d'urgence
        urgency = self._evaluate_urgency(claim)

        # Extraction d'entités
        entities = self._extract_entities(claim)

        # Mise à jour de l'état
        state['decomposed_assertions'] = assertions
        state['classification'] = {
            'theme': theme,
            'complexity': complexity,
            'urgency': urgency,
            'entities': entities,
            'num_assertions': len(assertions),
            'timestamp': datetime.now().isoformat()
        }

        # Trace de raisonnement
        if 'reasoning_trace' not in state:
            state['reasoning_trace'] = []

        state['reasoning_trace'].append(
            f"Classifier: Décomposé en {len(assertions)} assertions. "
            f"Thème: {theme}, Complexité: {complexity}/10"
        )

        return state

    def _decompose_claim(self, claim: str) -> List[str]:
        """
        Décompose une affirmation complexe en sous-assertions vérifiables.

        Args:
            claim: Affirmation à décomposer

        Returns:
            Liste d'assertions atomiques
        """
        if self.llm_client:
            return self._llm_decompose(claim)
        else:
            # Fallback: décomposition basique par phrases
            return self._basic_decompose(claim)

    def _llm_decompose(self, claim: str) -> List[str]:
        """
        Utilise un LLM pour décomposer l'affirmation.

        Args:
            claim: Affirmation à décomposer

        Returns:
            Liste d'assertions
        """
        prompt = f"""Décomposez cette affirmation en assertions atomiques vérifiables.

Affirmation: {claim}

Règles:
1. Chaque assertion doit être vérifiable indépendamment
2. Extraire les faits factuels (qui, quoi, quand, où)
3. Séparer les opinions des faits
4. Identifier les affirmations implicites

Répondez au format JSON:
{{
    "assertions": [
        "<assertion 1>",
        "<assertion 2>",
        ...
    ]
}}
"""

        try:
            if hasattr(self.llm_client, 'invoke'):
                response = self.llm_client.invoke(prompt)
            else:
                return self._basic_decompose(claim)

            if isinstance(response, str):
                result = json.loads(response)
            else:
                result = response

            return result.get('assertions', [claim])

        except Exception as e:
            print(f"Erreur décomposition LLM: {e}")
            return self._basic_decompose(claim)

    def _basic_decompose(self, claim: str) -> List[str]:
        """
        Décomposition basique sans LLM.

        Args:
            claim: Affirmation à décomposer

        Returns:
            Liste d'assertions (phrases séparées)
        """
        # Séparation simple par phrases
        import re
        sentences = re.split(r'[.!?]+', claim)
        assertions = [s.strip() for s in sentences if s.strip()]
        return assertions if assertions else [claim]

    def _classify_theme(self, claim: str) -> str:
        """
        Classifie le thème principal de l'affirmation.

        Args:
            claim: Affirmation à classifier

        Returns:
            Thème identifié
        """
        claim_lower = claim.lower()

        # Mapping de mots-clés vers thèmes
        theme_keywords = {
            'politique': ['gouvernement', 'président', 'ministre', 'élection', 'parti', 'vote'],
            'santé': ['covid', 'vaccin', 'maladie', 'médecin', 'hôpital', 'santé', 'traitement'],
            'climat': ['climat', 'réchauffement', 'environnement', 'pollution', 'co2', 'écologie'],
            'finance': ['économie', 'bourse', 'euro', 'dollar', 'inflation', 'banque', 'crise'],
            'technologie': ['ia', 'intelligence artificielle', 'tech', 'internet', 'data', 'cyber'],
            'science': ['recherche', 'étude', 'scientifique', 'université', 'laboratoire'],
            'société': ['social', 'société', 'culture', 'éducation', 'famille'],
            'international': ['guerre', 'pays', 'international', 'monde', 'conflit']
        }

        scores = {theme: 0 for theme in self.themes}

        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                if keyword in claim_lower:
                    scores[theme] += 1

        # Retourne le thème avec le score le plus élevé
        best_theme = max(scores, key=scores.get)

        return best_theme if scores[best_theme] > 0 else 'général'

    def _evaluate_complexity(self, claim: str, assertions: List[str]) -> int:
        """
        Évalue la complexité de l'affirmation (1-10).

        Args:
            claim: Affirmation originale
            assertions: Assertions décomposées

        Returns:
            Score de complexité (1-10)
        """
        score = 0

        # Facteur 1: Nombre d'assertions
        score += min(len(assertions), 5)

        # Facteur 2: Longueur
        if len(claim) > 200:
            score += 2
        elif len(claim) > 100:
            score += 1

        # Facteur 3: Mots complexes ou techniques
        complex_words = ['corrélation', 'causalité', 'statistique', 'étude', 'analyse']
        if any(word in claim.lower() for word in complex_words):
            score += 2

        # Facteur 4: Références temporelles multiples
        time_refs = ['2020', '2021', '2022', '2023', '2024', 'hier', 'demain']
        time_count = sum(1 for ref in time_refs if ref in claim.lower())
        score += min(time_count, 2)

        return min(score, 10)

    def _evaluate_urgency(self, claim: str) -> int:
        """
        Évalue l'urgence de vérification (1-10).

        Args:
            claim: Affirmation à évaluer

        Returns:
            Score d'urgence (1-10)
        """
        urgency = 5  # Baseline

        claim_lower = claim.lower()

        # Indicateurs d'urgence élevée
        urgent_keywords = ['urgent', 'alerte', 'danger', 'immédiat', 'maintenant', 'rapidement']
        if any(keyword in claim_lower for keyword in urgent_keywords):
            urgency += 3

        # Domaines critiques
        critical_domains = ['santé', 'sécurité', 'urgence', 'crise']
        if any(domain in claim_lower for domain in critical_domains):
            urgency += 2

        return min(urgency, 10)

    def _extract_entities(self, claim: str) -> Dict[str, List[str]]:
        """
        Extraction basique d'entités nommées.

        Args:
            claim: Affirmation à analyser

        Returns:
            Dict d'entités extraites
        """
        # Pour une version basique sans NER sophistiqué
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'numbers': []
        }

        # Détection basique de dates
        import re
        dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', claim)
        entities['dates'] = dates

        # Détection de nombres/pourcentages
        numbers = re.findall(r'\b\d+%|\b\d+\.\d+|\b\d+\b', claim)
        entities['numbers'] = numbers

        # Détection de noms propres (mots en majuscule)
        words = claim.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 1]

        # Classification simplifiée
        # (Dans un vrai système, on utiliserait spaCy ou un modèle NER)
        entities['persons'] = [w for w in capitalized if len(w) > 3]

        return entities
