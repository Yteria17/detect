"""
Orchestration Multi-Agents via LangGraph

Ce module implémente le workflow complet de fact-checking
en orchestrant les 5 agents spécialisés.
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from datetime import datetime
import operator

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph non disponible - utiliser: pip install langgraph")

from agents import (
    ClassifierAgent,
    RetrieverAgent,
    AnomalyDetectorAgent,
    FactCheckerAgent,
    ReporterAgent
)


class FactCheckingState(TypedDict):
    """État centralisé partagé par tous les agents."""

    # Input
    original_claim: str
    claim_id: Optional[str]

    # Classification
    decomposed_assertions: List[str]
    classification: Dict

    # Evidence
    evidence_retrieved: List[Dict]

    # Anomaly Detection
    anomaly_scores: Dict[str, float]
    anomaly_analysis: Dict

    # Fact Checking
    assertion_verdicts: Dict[str, Dict]

    # Final Results
    final_verdict: str
    confidence: float
    explanation: str

    # Reporting
    final_report: Dict

    # Metadata
    created_at: str
    agents_involved: List[str]
    reasoning_trace: Annotated[List[str], operator.add]  # Cumulative list


class MultiAgentFactChecker:
    """
    Orchestrateur principal du système multi-agents.
    Coordonne les 5 agents via un workflow LangGraph.
    """

    def __init__(
        self,
        llm_client=None,
        vector_store=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the Multi-Agent System.

        Args:
            llm_client: Client LLM (Claude, GPT-4, etc.)
            vector_store: Vector database pour RAG
            config: Configuration globale
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.config = config or {}

        # Initialisation des agents
        self.classifier = ClassifierAgent(llm_client, config.get('classifier', {}))
        self.retriever = RetrieverAgent(vector_store, config.get('retriever', {}))
        self.anomaly_detector = AnomalyDetectorAgent(llm_client, config.get('anomaly', {}))
        self.fact_checker = FactCheckerAgent(llm_client, config.get('fact_checker', {}))
        self.reporter = ReporterAgent(config.get('reporter', {}))

        # Compilation du workflow
        if LANGGRAPH_AVAILABLE:
            self.workflow = self._build_langgraph_workflow()
        else:
            self.workflow = None
            print("⚠️ Workflow manuel activé (LangGraph non disponible)")

    def _build_langgraph_workflow(self) -> StateGraph:
        """
        Construit le workflow LangGraph.

        Returns:
            Workflow compilé
        """
        # Créer le graphe d'état
        workflow = StateGraph(FactCheckingState)

        # Ajouter les nœuds (agents)
        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("anomaly_detector", self._anomaly_detector_node)
        workflow.add_node("fact_checker", self._fact_checker_node)
        workflow.add_node("reporter", self._reporter_node)

        # Définir les arêtes (flux d'exécution)
        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "retriever")
        workflow.add_edge("retriever", "anomaly_detector")
        workflow.add_edge("anomaly_detector", "fact_checker")
        workflow.add_edge("fact_checker", "reporter")
        workflow.add_edge("reporter", END)

        # Compiler le workflow
        return workflow.compile()

    def _classifier_node(self, state: FactCheckingState) -> FactCheckingState:
        """Node wrapper pour ClassifierAgent."""
        return self.classifier.classify(state)

    def _retriever_node(self, state: FactCheckingState) -> FactCheckingState:
        """Node wrapper pour RetrieverAgent."""
        return self.retriever.retrieve(state)

    def _anomaly_detector_node(self, state: FactCheckingState) -> FactCheckingState:
        """Node wrapper pour AnomalyDetectorAgent."""
        state = self.anomaly_detector.analyze(state)
        return self.anomaly_detector.escalate_if_needed(state)

    def _fact_checker_node(self, state: FactCheckingState) -> FactCheckingState:
        """Node wrapper pour FactCheckerAgent."""
        return self.fact_checker.verify(state)

    def _reporter_node(self, state: FactCheckingState) -> FactCheckingState:
        """Node wrapper pour ReporterAgent."""
        return self.reporter.generate_report(state)

    def check_claim(self, claim: str, claim_id: Optional[str] = None) -> Dict:
        """
        Vérifie une affirmation via le workflow multi-agents.

        Args:
            claim: Affirmation à vérifier
            claim_id: ID optionnel de la claim

        Returns:
            Rapport final de vérification
        """
        # État initial
        initial_state = FactCheckingState(
            original_claim=claim,
            claim_id=claim_id,
            decomposed_assertions=[],
            classification={},
            evidence_retrieved=[],
            anomaly_scores={},
            anomaly_analysis={},
            assertion_verdicts={},
            final_verdict='',
            confidence=0.0,
            explanation='',
            final_report={},
            created_at=datetime.now().isoformat(),
            agents_involved=[],
            reasoning_trace=[]
        )

        # Exécution du workflow
        if self.workflow:
            # Via LangGraph
            result = self.workflow.invoke(initial_state)
        else:
            # Exécution manuelle séquentielle
            result = self._execute_manual_workflow(initial_state)

        return result.get('final_report', result)

    def _execute_manual_workflow(self, state: FactCheckingState) -> FactCheckingState:
        """
        Exécution manuelle du workflow (fallback sans LangGraph).

        Args:
            state: État initial

        Returns:
            État final
        """
        print("🔄 Exécution workflow manuel...")

        # Séquence d'agents
        state = self.classifier.classify(state)
        print("  ✓ Classifier")

        state = self.retriever.retrieve(state)
        print("  ✓ Retriever")

        state = self.anomaly_detector.analyze(state)
        state = self.anomaly_detector.escalate_if_needed(state)
        print("  ✓ Anomaly Detector")

        state = self.fact_checker.verify(state)
        print("  ✓ Fact Checker")

        state = self.reporter.generate_report(state)
        print("  ✓ Reporter")

        print("✅ Workflow terminé")

        return state

    def check_multiple_claims(self, claims: List[str]) -> List[Dict]:
        """
        Vérifie plusieurs affirmations en parallèle.

        Args:
            claims: Liste d'affirmations

        Returns:
            Liste de rapports
        """
        results = []

        for i, claim in enumerate(claims):
            print(f"\n{'='*60}")
            print(f"Traitement claim {i+1}/{len(claims)}")
            print(f"{'='*60}")

            result = self.check_claim(claim, claim_id=f"claim_{i+1}")
            results.append(result)

        return results

    def export_report(
        self,
        report: Dict,
        filepath: str,
        format: str = 'json'
    ) -> None:
        """
        Exporte un rapport vers un fichier.

        Args:
            report: Rapport à exporter
            filepath: Chemin du fichier
            format: Format ('json' ou 'markdown')
        """
        state_with_report = {'final_report': report}

        if format == 'json':
            self.reporter.export_report_json(state_with_report, filepath)
            print(f"📄 Rapport JSON exporté: {filepath}")

        elif format == 'markdown':
            self.reporter.export_report_markdown(state_with_report, filepath)
            print(f"📝 Rapport Markdown exporté: {filepath}")

        else:
            raise ValueError(f"Format non supporté: {format}")

    def get_workflow_visualization(self) -> str:
        """
        Retourne une représentation textuelle du workflow.

        Returns:
            Diagramme ASCII du workflow
        """
        return """
┌─────────────────────────────────────────────────────┐
│         WORKFLOW MULTI-AGENTS FACT-CHECKING         │
└─────────────────────────────────────────────────────┘

                    [INPUT: Claim]
                           ↓
                  ┌────────────────┐
                  │   Classifier   │
                  │  (Décompose &  │
                  │   Classifie)   │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │   Retriever    │
                  │ (Collecte des  │
                  │    preuves)    │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │    Anomaly     │
                  │   Detector     │
                  │ (Détecte les   │
                  │   patterns)    │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │  Fact Checker  │
                  │  (Vérifie les  │
                  │   assertions)  │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │    Reporter    │
                  │  (Génère le    │
                  │   rapport)     │
                  └────────┬───────┘
                           ↓
                   [OUTPUT: Report]

Patterns supportés:
  • Pipeline séquentiel (actuel)
  • Parallel evidence retrieval
  • Consensus multi-path (avancé)
  • Dynamic escalation
        """


# Workflow alternatifs (Phase 3)

class ParallelMultiAgentFactChecker(MultiAgentFactChecker):
    """
    Variante avec récupération parallèle de preuves.
    """

    def _build_langgraph_workflow(self) -> StateGraph:
        """Workflow avec parallélisation."""
        if not LANGGRAPH_AVAILABLE:
            return None

        workflow = StateGraph(FactCheckingState)

        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("retriever_web", self._retriever_web_node)
        workflow.add_node("retriever_factcheck", self._retriever_factcheck_node)
        workflow.add_node("anomaly_detector", self._anomaly_detector_node)
        workflow.add_node("fact_checker", self._fact_checker_node)
        workflow.add_node("reporter", self._reporter_node)

        workflow.set_entry_point("classifier")

        # Parallélisation des retrievers
        workflow.add_edge("classifier", "retriever_web")
        workflow.add_edge("classifier", "retriever_factcheck")

        # Convergence vers anomaly detector
        workflow.add_edge("retriever_web", "anomaly_detector")
        workflow.add_edge("retriever_factcheck", "anomaly_detector")

        workflow.add_edge("anomaly_detector", "fact_checker")
        workflow.add_edge("fact_checker", "reporter")
        workflow.add_edge("reporter", END)

        return workflow.compile()

    def _retriever_web_node(self, state: FactCheckingState) -> FactCheckingState:
        """Retriever spécialisé web."""
        # Implémentation spécialisée
        return self.retriever.retrieve(state)

    def _retriever_factcheck_node(self, state: FactCheckingState) -> FactCheckingState:
        """Retriever spécialisé fact-checking databases."""
        # Recherche dans bases de fact-checking
        claims = state.get('decomposed_assertions', [])
        for claim in claims:
            results = self.retriever.search_fact_checking_databases(claim)
            state['evidence_retrieved'].extend(results)
        return state
