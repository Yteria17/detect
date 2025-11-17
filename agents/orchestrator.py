"""LangGraph orchestrator for multi-agent fact-checking workflow."""

from typing import Dict, Any
from datetime import datetime
import time
from langgraph.graph import StateGraph, END
from utils.types import FactCheckingState, Verdict, Priority
from utils.logger import log
from utils.helpers import generate_claim_id

# Import all agents
from agents.collector_agent import CollectorAgent
from agents.classifier_agent import ClassifierAgent
from agents.anomaly_detector_agent import AnomalyDetectorAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.reporter_agent import ReporterAgent


class FactCheckOrchestrator:
    """
    Orchestrates the multi-agent fact-checking workflow using LangGraph.

    Workflow:
    1. Classifier: Categorize and decompose claim
    2. Collector: Gather evidence from multiple sources
    3. Anomaly Detector: Identify suspicious patterns
    4. Fact Checker: Verify assertions against evidence
    5. Reporter: Consolidate and generate final report
    """

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        enable_logging: bool = True
    ):
        """
        Initialize the orchestrator.

        Args:
            llm_client: Language model client for agents
            retriever: RAG retriever for evidence collection
            enable_logging: Whether to enable detailed logging
        """
        self.enable_logging = enable_logging

        # Initialize all agents
        self.classifier = ClassifierAgent(llm_client=llm_client)
        self.collector = CollectorAgent()
        self.anomaly_detector = AnomalyDetectorAgent(llm_client=llm_client)
        self.fact_checker = FactCheckerAgent(llm_client=llm_client, retriever=retriever)
        self.reporter = ReporterAgent()

        # Build the workflow graph
        self.workflow = self._build_workflow()

        log.info("FactCheckOrchestrator initialized with all agents")

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(FactCheckingState)

        # Add agent nodes
        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("collector", self._collector_node)
        workflow.add_node("anomaly_detector", self._anomaly_detector_node)
        workflow.add_node("fact_checker", self._fact_checker_node)
        workflow.add_node("reporter", self._reporter_node)

        # Define the execution flow (sequential pipeline)
        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "collector")
        workflow.add_edge("collector", "anomaly_detector")
        workflow.add_edge("anomaly_detector", "fact_checker")
        workflow.add_edge("fact_checker", "reporter")
        workflow.add_edge("reporter", END)

        # Compile the graph
        return workflow.compile()

    async def _classifier_node(self, state: FactCheckingState) -> FactCheckingState:
        """Classifier agent node."""
        return await self.classifier.process(state)

    async def _collector_node(self, state: FactCheckingState) -> FactCheckingState:
        """Collector agent node."""
        return await self.collector.process(state)

    async def _anomaly_detector_node(self, state: FactCheckingState) -> FactCheckingState:
        """Anomaly detector agent node."""
        return await self.anomaly_detector.process(state)

    async def _fact_checker_node(self, state: FactCheckingState) -> FactCheckingState:
        """Fact checker agent node."""
        return await self.fact_checker.process(state)

    async def _reporter_node(self, state: FactCheckingState) -> FactCheckingState:
        """Reporter agent node."""
        return await self.reporter.process(state)

    async def check_claim(
        self,
        claim: str,
        priority: Priority = Priority.NORMAL,
        claim_id: str = None
    ) -> FactCheckingState:
        """
        Execute the full fact-checking workflow for a claim.

        Args:
            claim: Claim text to verify
            priority: Priority level
            claim_id: Optional custom claim ID

        Returns:
            Final state with verification results
        """
        start_time = time.time()

        # Create initial state
        initial_state = FactCheckingState(
            claim_id=claim_id or generate_claim_id(claim),
            original_claim=claim,
            priority=priority,
            created_at=datetime.now()
        )

        log.info(f"Starting fact-check workflow for claim: {initial_state.claim_id}")

        try:
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            final_state.processing_time_ms = processing_time_ms

            log.info(
                f"Fact-check completed for {final_state.claim_id} in {processing_time_ms}ms. "
                f"Verdict: {final_state.final_verdict.value} ({final_state.confidence:.2%})"
            )

            return final_state

        except Exception as e:
            log.error(f"Error during fact-check workflow: {str(e)}")
            raise

    def get_workflow_visualization(self) -> str:
        """
        Get ASCII visualization of the workflow graph.

        Returns:
            Workflow diagram
        """
        return """
┌─────────────────────────────────────────────────┐
│         Fact-Checking Multi-Agent Workflow       │
└─────────────────────────────────────────────────┘

                    ┌───────────┐
                    │   START   │
                    └─────┬─────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   CLASSIFIER    │
                 │  - Categorize    │
                 │  - Decompose     │
                 │  - Extract NER   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   COLLECTOR     │
                 │  - Web Search    │
                 │  - News APIs     │
                 │  - Social Media  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ ANOMALY DETECTOR │
                 │  - Patterns      │
                 │  - Coherence     │
                 │  - Red Flags     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  FACT CHECKER   │
                 │  - Verify        │
                 │  - RAG           │
                 │  - CoT Reasoning │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    REPORTER     │
                 │  - Consolidate   │
                 │  - Generate      │
                 │  - Escalate?     │
                 └────────┬─────────┘
                          │
                          ▼
                    ┌───────────┐
                    │    END    │
                    └───────────┘
"""


# Factory function for easy initialization
def create_orchestrator(
    llm_client=None,
    retriever=None,
    enable_logging: bool = True
) -> FactCheckOrchestrator:
    """
    Create a FactCheckOrchestrator instance.

    Args:
        llm_client: LLM client for agents
        retriever: RAG retriever
        enable_logging: Enable logging

    Returns:
        Initialized orchestrator
    """
    return FactCheckOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        enable_logging=enable_logging
    )
