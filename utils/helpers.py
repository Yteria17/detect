"""Helper functions for the fact-checking system."""

import hashlib
import re
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse
import unicodedata


def generate_claim_id(claim: str) -> str:
    """Generate a unique ID for a claim based on its content."""
    claim_normalized = claim.lower().strip()
    hash_object = hashlib.sha256(claim_normalized.encode())
    return hash_object.hexdigest()[:16]


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except Exception:
        return "unknown"


def normalize_text(text: str) -> str:
    """Normalize text for processing."""
    # Remove accents
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


def calculate_confidence_score(verdicts: List[Dict]) -> float:
    """
    Calculate overall confidence score from individual verdicts.

    Args:
        verdicts: List of verdict dictionaries with 'verdict' and 'confidence' keys

    Returns:
        Float between 0 and 1 representing overall confidence
    """
    if not verdicts:
        return 0.0

    # Weight by confidence scores
    total_weight = sum(v.get('confidence', 0.5) for v in verdicts)
    if total_weight == 0:
        return 0.0

    # Calculate weighted average
    weighted_score = sum(
        v.get('confidence', 0.5) * (1.0 if v.get('verdict') in ['SUPPORTED', 'REFUTED'] else 0.5)
        for v in verdicts
    ) / total_weight

    return min(max(weighted_score, 0.0), 1.0)


def merge_evidence(evidence_list: List[Dict]) -> List[Dict]:
    """
    Merge and deduplicate evidence from multiple sources.

    Args:
        evidence_list: List of evidence dictionaries

    Returns:
        Deduplicated and sorted evidence list
    """
    # Deduplicate by source URL
    seen_urls = set()
    merged = []

    for evidence in evidence_list:
        source_url = evidence.get('source', {}).get('url')
        if source_url and source_url not in seen_urls:
            seen_urls.add(source_url)
            merged.append(evidence)

    # Sort by relevance and credibility
    merged.sort(
        key=lambda x: (
            x.get('relevance_score', 0) * x.get('source', {}).get('credibility_score', 0)
        ),
        reverse=True
    )

    return merged


def format_reasoning_trace(trace: List[str]) -> str:
    """Format reasoning trace for human readability."""
    return '\n'.join(f"{i+1}. {step}" for i, step in enumerate(trace))


def estimate_complexity(claim: str) -> int:
    """
    Estimate claim complexity based on heuristics.

    Returns complexity score from 1 (simple) to 10 (very complex)
    """
    # Factors that increase complexity
    score = 1

    # Length
    word_count = len(claim.split())
    if word_count > 50:
        score += 2
    elif word_count > 30:
        score += 1

    # Number of entities (rough estimate)
    capital_words = len([w for w in claim.split() if w[0].isupper() and len(w) > 2])
    if capital_words > 5:
        score += 2
    elif capital_words > 3:
        score += 1

    # Numbers and statistics
    numbers = len(re.findall(r'\d+', claim))
    if numbers > 3:
        score += 1

    # Conditional/complex language
    complex_words = ['however', 'although', 'despite', 'unless', 'whereas', 'furthermore']
    if any(word in claim.lower() for word in complex_words):
        score += 1

    # URLs (suggests external references)
    if extract_urls(claim):
        score += 1

    return min(score, 10)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove non-alphanumeric characters except dots, dashes, and underscores
    sanitized = re.sub(r'[^\w\s\-\.]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    return sanitized.lower()


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
