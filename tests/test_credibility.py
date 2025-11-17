"""Tests for credibility scoring."""

import pytest
from utils.credibility import (
    score_source_credibility,
    score_unknown_source,
    get_credibility_label,
    adjust_credibility_by_context
)


def test_known_source_credibility():
    """Test credibility scoring for known sources."""
    # High credibility sources
    assert score_source_credibility("https://www.bbc.com/news/article") > 0.9
    assert score_source_credibility("https://www.reuters.com/article") > 0.9
    assert score_source_credibility("https://www.nature.com/article") > 0.9

    # Medium credibility
    assert 0.7 < score_source_credibility("https://www.wikipedia.org/wiki/Article") < 0.8

    # Low credibility (social media)
    assert score_source_credibility("https://twitter.com/user/status/123") < 0.4
    assert score_source_credibility("https://www.reddit.com/r/news/123") < 0.4


def test_unknown_source_heuristics():
    """Test heuristic scoring for unknown sources."""
    # .gov domains should score higher
    gov_score = score_unknown_source("example.gov")
    com_score = score_unknown_source("example.com")
    assert gov_score > com_score

    # .edu domains should score higher
    edu_score = score_unknown_source("example.edu")
    assert edu_score > com_score

    # Suspicious keywords should lower score
    normal_score = score_unknown_source("newssite.com")
    suspicious_score = score_unknown_source("fakenews.com")
    assert suspicious_score < normal_score


def test_credibility_labels():
    """Test credibility label assignment."""
    assert get_credibility_label(0.95) == "Very High"
    assert get_credibility_label(0.80) == "High"
    assert get_credibility_label(0.65) == "Medium-High"
    assert get_credibility_label(0.50) == "Medium"
    assert get_credibility_label(0.30) == "Low"
    assert get_credibility_label(0.10) == "Very Low"


def test_context_adjustment():
    """Test credibility adjustment based on context."""
    base_score = 0.5

    # With author and citations should increase
    adjusted = adjust_credibility_by_context(
        base_score,
        has_author=True,
        has_citations=True,
        is_recent=True,
        multiple_sources_confirm=True
    )
    assert adjusted > base_score

    # Without context should stay similar or decrease
    adjusted_low = adjust_credibility_by_context(
        base_score,
        has_author=False,
        has_citations=False,
        is_recent=False,
        multiple_sources_confirm=False
    )
    assert adjusted_low <= base_score


def test_domain_extraction():
    """Test domain extraction from URLs."""
    from utils.helpers import extract_domain

    assert extract_domain("https://www.bbc.com/news/article") == "bbc.com"
    assert extract_domain("http://subdomain.example.org/path") == "subdomain.example.org"
    assert extract_domain("https://www.twitter.com/user") == "twitter.com"
