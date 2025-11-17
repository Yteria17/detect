"""Tests for the classifier agent."""

import pytest
from agents.classifier_agent import ClassifierAgent
from utils.types import FactCheckingState, Priority
from utils.helpers import generate_claim_id


@pytest.fixture
def classifier():
    """Create a classifier agent instance."""
    return ClassifierAgent(llm_client=None)


@pytest.fixture
def sample_state():
    """Create a sample fact-checking state."""
    claim = "The president announced new climate policies yesterday"
    return FactCheckingState(
        claim_id=generate_claim_id(claim),
        original_claim=claim,
        priority=Priority.NORMAL
    )


@pytest.mark.asyncio
async def test_classifier_process(classifier, sample_state):
    """Test that classifier processes a claim successfully."""
    result = await classifier.process(sample_state)

    assert result.classification is not None
    assert result.classification.theme in classifier.themes or result.classification.theme == 'other'
    assert 1 <= result.classification.complexity <= 10
    assert 1 <= result.classification.urgency <= 10
    assert len(result.decomposed_assertions) > 0
    assert len(result.reasoning_trace) > 0


@pytest.mark.asyncio
async def test_theme_detection(classifier):
    """Test theme detection for various claims."""
    test_cases = [
        ("The vaccine prevents COVID-19 infection", "health"),
        ("The president won the election", "politics"),
        ("Global temperatures are rising rapidly", "climate"),
        ("Scientists discovered a new particle", "science"),
    ]

    for claim, expected_theme in test_cases:
        detected_theme = classifier._detect_theme_keywords(claim.lower())
        assert detected_theme == expected_theme


def test_complexity_estimation(classifier):
    """Test complexity scoring."""
    simple_claim = "Paris is in France."
    complex_claim = "The president, who was elected in 2020 despite multiple controversies, announced yesterday that new climate policies would be implemented, although several scientists have raised concerns about the proposed 150% reduction target."

    classification_simple = classifier._heuristic_classify(simple_claim)
    classification_complex = classifier._heuristic_classify(complex_claim)

    assert classification_simple.complexity < classification_complex.complexity


@pytest.mark.asyncio
async def test_assertion_decomposition(classifier):
    """Test assertion decomposition."""
    simple_claim = "Paris is the capital of France."
    complex_claim = "The president announced policies yesterday and scientists support them."

    state_simple = FactCheckingState(
        claim_id="test1",
        original_claim=simple_claim,
        priority=Priority.NORMAL
    )

    state_complex = FactCheckingState(
        claim_id="test2",
        original_claim=complex_claim,
        priority=Priority.NORMAL
    )

    result_simple = await classifier.process(state_simple)
    result_complex = await classifier.process(state_complex)

    # Simple claim should have 1 assertion
    assert len(result_simple.decomposed_assertions) == 1

    # Complex claim may have multiple (depending on decomposition)
    assert len(result_complex.decomposed_assertions) >= 1
