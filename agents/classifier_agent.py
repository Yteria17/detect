"""Agent 2: Classifier - Categorizes claims and extracts entities."""

from typing import List, Dict
import json
from agents.base_agent import BaseAgent
from utils.types import FactCheckingState, Classification, Assertion
from utils.logger import log
from utils.helpers import generate_claim_id, estimate_complexity
from config.settings import settings


class ClassifierAgent(BaseAgent):
    """
    Classifier Agent.

    Responsibilities:
    - Categorize claims by theme/domain
    - Extract named entities
    - Decompose complex claims into verifiable assertions
    - Assess claim complexity and urgency
    """

    def __init__(self, llm_client=None):
        super().__init__(
            name="ClassifierAgent",
            description="Categorizes claims and decomposes them into verifiable assertions"
        )
        self.llm = llm_client
        # Theme categories
        self.themes = [
            "politics", "health", "climate", "science", "economy",
            "technology", "sports", "entertainment", "other"
        ]

    async def process(self, state: FactCheckingState) -> FactCheckingState:
        """
        Classify claim and decompose into assertions.

        Args:
            state: Current fact-checking state

        Returns:
            Updated state with classification and assertions
        """
        self.log_action(state, "Starting claim classification and decomposition")

        claim = state.original_claim

        # Step 1: Classify theme and extract entities
        classification = await self._classify_claim(claim)
        state.classification = classification

        self.log_action(
            state,
            f"Classified claim as '{classification.theme}' with complexity {classification.complexity}/10"
        )

        # Step 2: Decompose into assertions
        assertions = await self._decompose_claim(claim, classification)
        state.decomposed_assertions = assertions

        self.log_action(
            state,
            f"Decomposed claim into {len(assertions)} verifiable assertions"
        )

        return state

    async def _classify_claim(self, claim: str) -> Classification:
        """
        Classify claim by theme and extract metadata.

        Args:
            claim: Claim text

        Returns:
            Classification object
        """
        if self.llm:
            # Use LLM for classification
            return await self._llm_classify(claim)
        else:
            # Fallback to heuristic classification
            return self._heuristic_classify(claim)

    async def _llm_classify(self, claim: str) -> Classification:
        """
        Use LLM to classify claim.

        Args:
            claim: Claim text

        Returns:
            Classification object
        """
        prompt = f"""Analyze this claim and provide classification:

Claim: "{claim}"

Classify according to:
1. Theme: {', '.join(self.themes)}
2. Complexity (1-10): How complex is this claim to verify?
3. Urgency (1-10): How urgent is verification? (misinformation risk, viral potential)
4. Requires multimodal: Does this need image/video/audio analysis?
5. Detected entities: List key entities (people, organizations, places, dates)

Respond in JSON format:
{{
  "theme": "...",
  "complexity": 5,
  "urgency": 7,
  "requires_multimodal": false,
  "detected_entities": ["entity1", "entity2", ...]
}}
"""

        try:
            # TODO: Replace with actual LLM call
            # response = await self.llm.ainvoke(prompt)
            # result = json.loads(response.content)

            # Placeholder response
            result = {
                "theme": self._detect_theme_keywords(claim),
                "complexity": estimate_complexity(claim),
                "urgency": 5,
                "requires_multimodal": False,
                "detected_entities": []
            }

            return Classification(**result)

        except Exception as e:
            log.error(f"[{self.name}] LLM classification failed: {str(e)}")
            return self._heuristic_classify(claim)

    def _heuristic_classify(self, claim: str) -> Classification:
        """
        Fallback heuristic classification.

        Args:
            claim: Claim text

        Returns:
            Classification object
        """
        claim_lower = claim.lower()

        # Detect theme by keywords
        theme = self._detect_theme_keywords(claim_lower)

        # Estimate complexity
        complexity = estimate_complexity(claim)

        # Estimate urgency based on keywords
        urgency_keywords = {
            'urgent': 8, 'breaking': 9, 'alert': 8, 'warning': 7,
            'confirmed': 6, 'official': 5, 'rumor': 7, 'viral': 8
        }
        urgency = max(
            [score for keyword, score in urgency_keywords.items() if keyword in claim_lower],
            default=5
        )

        # Check if multimodal
        requires_multimodal = any(
            keyword in claim_lower
            for keyword in ['video', 'image', 'photo', 'footage', 'audio', 'recording']
        )

        # Simple entity extraction (capitalized words)
        words = claim.split()
        detected_entities = [
            word.strip('.,!?')
            for word in words
            if word and word[0].isupper() and len(word) > 2
        ][:10]  # Limit to 10

        return Classification(
            theme=theme,
            complexity=complexity,
            urgency=urgency,
            requires_multimodal=requires_multimodal,
            detected_entities=detected_entities
        )

    def _detect_theme_keywords(self, claim: str) -> str:
        """
        Detect theme using keyword matching.

        Args:
            claim: Claim text (lowercase)

        Returns:
            Detected theme
        """
        theme_keywords = {
            'politics': ['president', 'election', 'government', 'minister', 'vote', 'parliament', 'senator'],
            'health': ['vaccine', 'covid', 'disease', 'virus', 'health', 'medical', 'doctor', 'hospital'],
            'climate': ['climate', 'warming', 'temperature', 'carbon', 'emission', 'renewable', 'fossil'],
            'science': ['study', 'research', 'scientist', 'university', 'published', 'journal'],
            'economy': ['economy', 'market', 'stock', 'inflation', 'gdp', 'unemployment', 'financial'],
            'technology': ['ai', 'technology', 'software', 'app', 'digital', 'cyber', 'internet'],
            'sports': ['football', 'basketball', 'olympics', 'champion', 'match', 'player', 'team'],
            'entertainment': ['movie', 'actor', 'celebrity', 'film', 'music', 'artist', 'concert']
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in claim for keyword in keywords):
                return theme

        return 'other'

    async def _decompose_claim(self, claim: str, classification: Classification) -> List[Assertion]:
        """
        Decompose claim into verifiable assertions.

        Args:
            claim: Claim text
            classification: Claim classification

        Returns:
            List of assertions
        """
        if self.llm and classification.complexity > 5:
            # Use LLM for complex claims
            return await self._llm_decompose(claim)
        else:
            # For simple claims, treat as single assertion
            return self._simple_decompose(claim)

    async def _llm_decompose(self, claim: str) -> List[Assertion]:
        """
        Use LLM to decompose claim into assertions.

        Args:
            claim: Claim text

        Returns:
            List of assertions
        """
        prompt = f"""Decompose this claim into individual, verifiable assertions:

Claim: "{claim}"

Break it down into atomic facts that can each be independently verified.
Each assertion should be:
- A single factual statement
- Independently verifiable
- Clear and specific

Respond in JSON format:
{{
  "assertions": [
    "assertion 1",
    "assertion 2",
    ...
  ]
}}
"""

        try:
            # TODO: Replace with actual LLM call
            # response = await self.llm.ainvoke(prompt)
            # result = json.loads(response.content)
            # assertions_text = result['assertions']

            # Placeholder - split by sentence
            assertions_text = self._split_into_sentences(claim)

        except Exception as e:
            log.error(f"[{self.name}] LLM decomposition failed: {str(e)}")
            assertions_text = self._split_into_sentences(claim)

        # Convert to Assertion objects
        assertions = []
        for i, text in enumerate(assertions_text):
            assertion = Assertion(
                text=text.strip(),
                assertion_id=f"{generate_claim_id(claim)}_{i}"
            )
            assertions.append(assertion)

        return assertions

    def _simple_decompose(self, claim: str) -> List[Assertion]:
        """
        Simple decomposition for straightforward claims.

        Args:
            claim: Claim text

        Returns:
            List with single assertion
        """
        return [
            Assertion(
                text=claim.strip(),
                assertion_id=generate_claim_id(claim) + "_0"
            )
        ]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        import re
        # Simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
