"""Agent 3: Anomaly Detector - Detects semantic anomalies and inconsistencies."""

from typing import Dict, List
import json
from agents.base_agent import BaseAgent
from utils.types import FactCheckingState, AnomalyScore, Verdict
from utils.logger import log
from config.settings import settings


class AnomalyDetectorAgent(BaseAgent):
    """
    Anomaly Detector Agent.

    Responsibilities:
    - Detect semantic inconsistencies
    - Identify manipulative language patterns
    - Check logical coherence
    - Flag suspicious claims for deeper verification
    """

    def __init__(self, llm_client=None):
        super().__init__(
            name="AnomalyDetectorAgent",
            description="Detects semantic anomalies and suspicious patterns in claims"
        )
        self.llm = llm_client

        # Suspicious patterns to detect
        self.manipulation_patterns = [
            "100%", "absolutely", "guarantee", "secret", "they don't want you to know",
            "shocking", "urgent", "breaking", "confirmed", "proven fact",
            "scientists discovered", "doctors hate", "one simple trick"
        ]

        self.fear_keywords = [
            "danger", "warning", "threat", "risk", "deadly", "fatal",
            "crisis", "disaster", "catastrophe", "emergency"
        ]

    async def process(self, state: FactCheckingState) -> FactCheckingState:
        """
        Detect anomalies in decomposed assertions.

        Args:
            state: Current fact-checking state

        Returns:
            Updated state with anomaly scores
        """
        self.log_action(state, "Starting anomaly detection")

        if not state.decomposed_assertions:
            log.warning(f"[{self.name}] No assertions to analyze")
            return state

        # Analyze each assertion for anomalies
        for assertion in state.decomposed_assertions:
            anomaly_score = await self._detect_anomalies(assertion.text, state)
            state.anomaly_scores[assertion.assertion_id] = anomaly_score

        # Calculate average anomaly score
        avg_score = sum(s.score for s in state.anomaly_scores.values()) / len(state.anomaly_scores)

        high_anomaly_count = sum(1 for s in state.anomaly_scores.values() if s.score > 0.6)

        self.log_action(
            state,
            f"Detected {high_anomaly_count} high-anomaly assertions (avg score: {avg_score:.2f})"
        )

        return state

    async def _detect_anomalies(self, assertion: str, state: FactCheckingState) -> AnomalyScore:
        """
        Detect anomalies in a single assertion.

        Args:
            assertion: Assertion text
            state: Current state

        Returns:
            Anomaly score object
        """
        detected_patterns = []
        score = 0.0

        # Pattern 1: Manipulative language
        manipulation_score = self._check_manipulation_patterns(assertion)
        if manipulation_score > 0:
            score += manipulation_score * 0.3
            detected_patterns.append("manipulative_language")

        # Pattern 2: Emotional appeal (fear, anger)
        emotion_score = self._check_emotional_appeal(assertion)
        if emotion_score > 0:
            score += emotion_score * 0.2
            detected_patterns.append("emotional_appeal")

        # Pattern 3: Absolute certainty claims
        certainty_score = self._check_absolute_certainty(assertion)
        if certainty_score > 0:
            score += certainty_score * 0.2
            detected_patterns.append("absolute_certainty")

        # Pattern 4: Lack of sources/attribution
        source_score = self._check_source_attribution(assertion)
        if source_score > 0:
            score += source_score * 0.15
            detected_patterns.append("lacks_attribution")

        # Pattern 5: LLM-based coherence check (if available)
        if self.llm:
            coherence_score = await self._llm_coherence_check(assertion, state)
            score += coherence_score * 0.15
            if coherence_score > 0.5:
                detected_patterns.append("logical_inconsistency")

        # Normalize score to 0-1
        score = min(score, 1.0)

        # Generate reasoning
        reasoning = self._generate_reasoning(assertion, score, detected_patterns)

        return AnomalyScore(
            assertion_id=assertion,
            score=score,
            detected_patterns=detected_patterns,
            reasoning=reasoning
        )

    def _check_manipulation_patterns(self, text: str) -> float:
        """
        Check for manipulative language patterns.

        Args:
            text: Text to analyze

        Returns:
            Score 0-1
        """
        text_lower = text.lower()
        matches = sum(1 for pattern in self.manipulation_patterns if pattern in text_lower)

        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.4
        elif matches == 2:
            return 0.7
        else:
            return 1.0

    def _check_emotional_appeal(self, text: str) -> float:
        """
        Check for emotional manipulation (fear, anger, urgency).

        Args:
            text: Text to analyze

        Returns:
            Score 0-1
        """
        text_lower = text.lower()

        # Count fear keywords
        fear_count = sum(1 for keyword in self.fear_keywords if keyword in text_lower)

        # Check for excessive exclamation marks
        exclamation_count = text.count('!')

        # Check for all caps words (shouting)
        words = text.split()
        caps_words = sum(1 for word in words if len(word) > 3 and word.isupper())

        score = 0.0
        if fear_count > 0:
            score += min(fear_count * 0.2, 0.5)
        if exclamation_count > 1:
            score += min(exclamation_count * 0.1, 0.3)
        if caps_words > 0:
            score += min(caps_words * 0.1, 0.2)

        return min(score, 1.0)

    def _check_absolute_certainty(self, text: str) -> float:
        """
        Check for absolute certainty claims without nuance.

        Args:
            text: Text to analyze

        Returns:
            Score 0-1
        """
        absolute_words = [
            "always", "never", "all", "none", "everyone", "nobody",
            "100%", "absolutely", "definitely", "certainly", "impossible",
            "proven", "undeniable", "irrefutable"
        ]

        text_lower = text.lower()
        matches = sum(1 for word in absolute_words if word in text_lower)

        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.3
        elif matches == 2:
            return 0.6
        else:
            return 0.9

    def _check_source_attribution(self, text: str) -> float:
        """
        Check if claim lacks proper source attribution.

        Args:
            text: Text to analyze

        Returns:
            Score 0-1 (higher = more suspicious due to lack of sources)
        """
        # Look for vague attributions
        vague_attributions = [
            "they say", "people say", "experts claim", "sources say",
            "it is said", "many believe", "some say", "rumor has it"
        ]

        text_lower = text.lower()

        # Check for vague attributions
        has_vague = any(attr in text_lower for attr in vague_attributions)

        # Check for specific sources (good sign)
        has_specific = any(
            marker in text_lower
            for marker in ["according to", "published in", "stated by", "reported by"]
        )

        if has_vague and not has_specific:
            return 0.7
        elif not has_specific and len(text.split()) > 15:  # Long claim without attribution
            return 0.4
        else:
            return 0.0

    async def _llm_coherence_check(self, assertion: str, state: FactCheckingState) -> float:
        """
        Use LLM to check logical coherence and consistency.

        Args:
            assertion: Assertion to check
            state: Current state (for context)

        Returns:
            Incoherence score 0-1
        """
        prompt = f"""Analyze this assertion for logical coherence and consistency:

Assertion: "{assertion}"

Check for:
1. Internal contradictions
2. Logical fallacies
3. Vague or ambiguous claims
4. Inconsistency with commonly known facts

Rate the suspicion level from 0.0 (completely coherent) to 1.0 (highly suspicious).

Respond with just a number between 0.0 and 1.0.
"""

        try:
            # TODO: Replace with actual LLM call
            # response = await self.llm.ainvoke(prompt)
            # score = float(response.content.strip())
            # return min(max(score, 0.0), 1.0)

            # Placeholder
            return 0.0

        except Exception as e:
            log.error(f"[{self.name}] LLM coherence check failed: {str(e)}")
            return 0.0

    def _generate_reasoning(self, assertion: str, score: float, patterns: List[str]) -> str:
        """
        Generate human-readable reasoning for anomaly score.

        Args:
            assertion: Analyzed assertion
            score: Anomaly score
            patterns: Detected patterns

        Returns:
            Reasoning text
        """
        if score < 0.3:
            return "Assertion appears normal with no significant suspicious patterns."
        elif score < 0.6:
            return f"Assertion shows some suspicious patterns: {', '.join(patterns)}. Moderate anomaly detected."
        else:
            return f"Assertion shows multiple red flags: {', '.join(patterns)}. High anomaly score suggests careful verification needed."
