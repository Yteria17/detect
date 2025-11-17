"""Multi-agent system for fact-checking."""

from agents.reporter import ReporterAgent, VerdictType, AlertLevel, FactCheckReport, Alert

__all__ = [
    "ReporterAgent",
    "VerdictType",
    "AlertLevel",
    "FactCheckReport",
    "Alert",
]
"""
Agents pour le système multi-agents de détection de désinformation.
"""

from .anomaly_detector import AnomalyDetectorAgent
from .classifier import ClassifierAgent
from .retriever import RetrieverAgent
from .fact_checker import FactCheckerAgent
from .reporter import ReporterAgent

__all__ = [
    'AnomalyDetectorAgent',
    'ClassifierAgent',
    'RetrieverAgent',
    'FactCheckerAgent',
    'ReporterAgent'
]
"""Multi-agent system for misinformation detection."""
