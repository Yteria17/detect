"""
Système Multi-Agents pour la Détection de Désinformation
Phase 2 - Implémentation

Ce package contient:
- 5 agents spécialisés
- Orchestration LangGraph
- Détection de deepfakes
- RAG hybride (BM25 + Semantic)
"""

__version__ = "0.2.0"
__author__ = "Detect Team"

from .workflow import MultiAgentFactChecker, ParallelMultiAgentFactChecker
from .agents import (
    ClassifierAgent,
    RetrieverAgent,
    AnomalyDetectorAgent,
    FactCheckerAgent,
    ReporterAgent
)
from .utils import DeepfakeDetector

__all__ = [
    'MultiAgentFactChecker',
    'ParallelMultiAgentFactChecker',
    'ClassifierAgent',
    'RetrieverAgent',
    'AnomalyDetectorAgent',
    'FactCheckerAgent',
    'ReporterAgent',
    'DeepfakeDetector'
]
