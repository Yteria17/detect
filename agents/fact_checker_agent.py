"""Agent 4: Fact Checker - Verifies claims against evidence using RAG."""

from typing import List, Dict, Optional
import json
from agents.base_agent import BaseAgent
from utils.types import FactCheckingState, Verdict, Evidence
from utils.logger import log
from utils.credibility import score_source_credibility
from config.settings import settings


class FactCheckerAgent(BaseAgent):
    """
    Fact Checker Agent.

    Responsibilities:
    - Cross-reference claims with collected evidence
    - Query fact-checking databases
    - Apply Chain-of-Thought reasoning for verification
    - Score claim veracity with confidence
    - Handle conflicting evidence
    """

    def __init__(self, llm_client=None, retriever=None):
        super().__init__(
            name="FactCheckerAgent",
            description="Verifies claims using evidence and reasoning"
        )
        self.llm = llm_client
        self.retriever = retriever  # RAG retriever

        # Known fact-checking databases (would be queried via API/scraping)
        self.factcheck_sources = [
            "snopes.com",
            "politifact.com",
            "factcheck.org",
            "fullfact.org",
            "factuel.afp.com",
            "checknews.fr"
        ]

    async def process(self, state: FactCheckingState) -> FactCheckingState:
        """
        Verify all assertions using evidence.

        Args:
            state: Current fact-checking state

        Returns:
            Updated state with verification results
        """
        self.log_action(state, "Starting fact verification")

        if not state.decomposed_assertions:
            log.warning(f"[{self.name}] No assertions to verify")
            return state

        # Verify each assertion
        for assertion in state.decomposed_assertions:
            # Get relevant evidence for this assertion
            relevant_evidence = self._filter_relevant_evidence(
                assertion.text,
                state.evidence_retrieved
            )

            # Retrieve additional evidence if needed
            if len(relevant_evidence) < 3 and self.retriever:
                additional = await self._retrieve_additional_evidence(assertion.text)
                relevant_evidence.extend(additional)

            # Verify assertion
            verdict = await self._verify_assertion(
                assertion.text,
                relevant_evidence,
                state
            )

            # Update assertion
            assertion.verdict = verdict['verdict']
            assertion.confidence = verdict['confidence']
            assertion.evidence = relevant_evidence

            # Store in triplet verdicts
            state.triplet_verdicts[assertion.assertion_id] = verdict

        verified_count = len([a for a in state.decomposed_assertions if a.verdict != Verdict.PENDING])

        self.log_action(
            state,
            f"Verified {verified_count}/{len(state.decomposed_assertions)} assertions"
        )

        return state

    def _filter_relevant_evidence(
        self,
        assertion: str,
        all_evidence: List[Evidence]
    ) -> List[Evidence]:
        """
        Filter evidence relevant to specific assertion.

        Args:
            assertion: Assertion text
            all_evidence: All collected evidence

        Returns:
            Filtered evidence list
        """
        # TODO: Use semantic similarity for better filtering
        # For now, simple keyword matching
        assertion_lower = assertion.lower()
        assertion_words = set(assertion_lower.split())

        relevant = []
        for evidence in all_evidence:
            evidence_words = set(evidence.text.lower().split())

            # Calculate simple word overlap
            overlap = len(assertion_words & evidence_words)
            if overlap >= 2:  # At least 2 common words
                evidence.relevance_score = overlap / len(assertion_words)
                relevant.append(evidence)

        # Sort by relevance and credibility
        relevant.sort(
            key=lambda e: e.relevance_score * e.source.credibility_score,
            reverse=True
        )

        return relevant[:settings.max_evidence_sources]

    async def _retrieve_additional_evidence(self, query: str) -> List[Evidence]:
        """
        Retrieve additional evidence using RAG retriever.

        Args:
            query: Search query

        Returns:
            Additional evidence
        """
        # TODO: Implement RAG retrieval
        # if self.retriever:
        #     docs = await self.retriever.aget_relevant_documents(query)
        #     return [self._convert_doc_to_evidence(doc) for doc in docs]

        return []

    async def _verify_assertion(
        self,
        assertion: str,
        evidence: List[Evidence],
        state: FactCheckingState
    ) -> Dict:
        """
        Verify a single assertion using Chain-of-Thought reasoning.

        Args:
            assertion: Assertion to verify
            evidence: Relevant evidence
            state: Current state

        Returns:
            Verdict dictionary with verdict, confidence, and reasoning
        """
        if not evidence:
            return {
                'verdict': Verdict.INSUFFICIENT_INFO,
                'confidence': 0.0,
                'reasoning': 'No evidence available for verification'
            }

        # Use LLM for verification if available
        if self.llm:
            return await self._llm_verify(assertion, evidence, state)
        else:
            return self._heuristic_verify(assertion, evidence)

    async def _llm_verify(
        self,
        assertion: str,
        evidence: List[Evidence],
        state: FactCheckingState
    ) -> Dict:
        """
        Use LLM with Chain-of-Thought to verify assertion.

        Args:
            assertion: Assertion to verify
            evidence: Evidence list
            state: Current state

        Returns:
            Verification result
        """
        # Prepare evidence for prompt
        evidence_text = "\n\n".join([
            f"[Source: {e.source.domain} (credibility: {e.source.credibility_score:.2f})]\n{e.text}"
            for e in evidence[:5]  # Top 5 pieces
        ])

        prompt = f"""You are an expert fact-checker. Verify this assertion using the provided evidence.

Assertion to verify: "{assertion}"

Available Evidence:
{evidence_text}

Follow these steps:
1. Identify the key factual claims in the assertion
2. For each claim, find supporting or contradicting evidence
3. Evaluate the credibility of each evidence source
4. Resolve any conflicting evidence
5. Determine final verdict

Respond in JSON format:
{{
  "reasoning_steps": ["step 1", "step 2", ...],
  "verdict": "SUPPORTED" | "REFUTED" | "INSUFFICIENT_INFO",
  "confidence": 0.0-1.0,
  "explanation": "detailed explanation"
}}

Guidelines:
- SUPPORTED: Strong evidence supports the assertion
- REFUTED: Strong evidence contradicts the assertion
- INSUFFICIENT_INFO: Evidence is lacking, unclear, or conflicting
- Weight evidence by source credibility
- Require high-credibility sources for strong claims
"""

        try:
            # TODO: Replace with actual LLM call
            # response = await self.llm.ainvoke(prompt)
            # result = json.loads(response.content)

            # Placeholder - use heuristic
            result = self._heuristic_verify(assertion, evidence)

            return result

        except Exception as e:
            log.error(f"[{self.name}] LLM verification failed: {str(e)}")
            return self._heuristic_verify(assertion, evidence)

    def _heuristic_verify(self, assertion: str, evidence: List[Evidence]) -> Dict:
        """
        Heuristic verification when LLM is not available.

        Args:
            assertion: Assertion to verify
            evidence: Evidence list

        Returns:
            Verification result
        """
        if not evidence:
            return {
                'verdict': Verdict.INSUFFICIENT_INFO,
                'confidence': 0.0,
                'reasoning': ['No evidence available']
            }

        # Simple heuristic: check if high-credibility sources support
        high_cred_evidence = [e for e in evidence if e.source.credibility_score > 0.7]

        if high_cred_evidence:
            # Assume supported if high-credibility sources present
            avg_credibility = sum(e.source.credibility_score for e in high_cred_evidence) / len(high_cred_evidence)

            return {
                'verdict': Verdict.SUPPORTED,
                'confidence': avg_credibility,
                'reasoning': [
                    f"Found {len(high_cred_evidence)} high-credibility sources",
                    f"Average source credibility: {avg_credibility:.2f}"
                ]
            }
        else:
            # Low credibility sources only
            return {
                'verdict': Verdict.INSUFFICIENT_INFO,
                'confidence': 0.4,
                'reasoning': [
                    f"Only low-credibility sources available ({len(evidence)} sources)",
                    "Cannot verify with confidence"
                ]
            }

    async def _query_factcheck_databases(self, assertion: str) -> List[Evidence]:
        """
        Query known fact-checking databases.

        Args:
            assertion: Assertion to check

        Returns:
            Evidence from fact-checking sources
        """
        # TODO: Implement actual API calls or web scraping
        # For Snopes, PolitiFact, etc.

        return []

    def _resolve_conflicting_evidence(self, evidence: List[Evidence]) -> Dict:
        """
        Resolve conflicting evidence using credibility weighting.

        Args:
            evidence: List of potentially conflicting evidence

        Returns:
            Resolution result
        """
        # Group evidence by sentiment (supporting vs refuting)
        # Weight by credibility
        # Return weighted verdict

        supporting_weight = 0.0
        refuting_weight = 0.0

        for e in evidence:
            # TODO: Classify evidence as supporting/refuting
            # For now, just accumulate credibility
            supporting_weight += e.source.credibility_score

        if supporting_weight > refuting_weight * 1.5:
            return {'verdict': Verdict.SUPPORTED, 'confidence': 0.7}
        elif refuting_weight > supporting_weight * 1.5:
            return {'verdict': Verdict.REFUTED, 'confidence': 0.7}
        else:
            return {'verdict': Verdict.CONFLICTING, 'confidence': 0.5}
