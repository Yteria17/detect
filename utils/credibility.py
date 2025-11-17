"""Source credibility scoring utilities."""

from typing import Dict, Optional
from urllib.parse import urlparse


# Pre-defined credibility scores for known sources
SOURCE_CREDIBILITY_DB: Dict[str, float] = {
    # High credibility news sources
    "bbc.com": 0.95,
    "bbc.co.uk": 0.95,
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "afp.com": 0.94,
    "lemonde.fr": 0.93,
    "lefigaro.fr": 0.92,
    "liberation.fr": 0.92,
    "theguardian.com": 0.93,
    "nytimes.com": 0.92,
    "washingtonpost.com": 0.91,
    "ft.com": 0.93,

    # Scientific sources
    "nature.com": 0.98,
    "science.org": 0.98,
    "cell.com": 0.97,
    "thelancet.com": 0.97,
    "nejm.org": 0.97,
    "pnas.org": 0.96,
    "arxiv.org": 0.85,  # Pre-prints, not peer-reviewed

    # Fact-checking organizations
    "snopes.com": 0.96,
    "politifact.com": 0.95,
    "factcheck.org": 0.95,
    "fullfact.org": 0.94,
    "checknews.fr": 0.93,
    "factuel.afp.com": 0.94,

    # Government/institutional
    "data.gouv.fr": 0.94,
    "europa.eu": 0.93,
    "un.org": 0.93,
    "who.int": 0.94,
    "cdc.gov": 0.94,

    # Wikipedia (useful but varies by article)
    "wikipedia.org": 0.75,
    "en.wikipedia.org": 0.75,
    "fr.wikipedia.org": 0.75,

    # Academic institutions
    "mit.edu": 0.92,
    "stanford.edu": 0.92,
    "ox.ac.uk": 0.92,
    "cam.ac.uk": 0.92,

    # Social media (lower credibility - requires verification)
    "twitter.com": 0.30,
    "x.com": 0.30,
    "facebook.com": 0.30,
    "reddit.com": 0.35,
    "youtube.com": 0.40,
    "tiktok.com": 0.25,

    # Known problematic sources
    "infowars.com": 0.10,
    "naturalnews.com": 0.15,
    "beforeitsnews.com": 0.15,
}


def score_source_credibility(url: str, domain: Optional[str] = None) -> float:
    """
    Score the credibility of a source based on its domain.

    Args:
        url: Source URL
        domain: Optional pre-extracted domain

    Returns:
        Credibility score between 0.0 (not credible) and 1.0 (highly credible)
    """
    if not domain:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '').lower()
        except Exception:
            return 0.5  # Default neutral score

    # Check exact match
    if domain in SOURCE_CREDIBILITY_DB:
        return SOURCE_CREDIBILITY_DB[domain]

    # Check for partial matches (e.g., subdomain.bbc.com)
    for known_domain, score in SOURCE_CREDIBILITY_DB.items():
        if domain.endswith(known_domain):
            return score * 0.9  # Slightly lower for subdomains

    # Heuristic scoring for unknown sources
    return score_unknown_source(domain)


def score_unknown_source(domain: str) -> float:
    """
    Heuristic scoring for unknown sources based on domain characteristics.

    Args:
        domain: Domain name

    Returns:
        Estimated credibility score
    """
    score = 0.5  # Start neutral

    # TLD-based adjustments
    if domain.endswith('.gov'):
        score += 0.2
    elif domain.endswith('.edu'):
        score += 0.15
    elif domain.endswith('.org'):
        score += 0.05
    elif domain.endswith(('.com', '.net')):
        score += 0.0
    elif domain.endswith(('.info', '.xyz', '.tk', '.ml')):
        score -= 0.1  # Often used for spam

    # Suspicious patterns
    suspicious_keywords = ['fake', 'hoax', 'conspiracy', 'secret', 'truth', 'exposed']
    if any(keyword in domain.lower() for keyword in suspicious_keywords):
        score -= 0.2

    # Hyphenation (often used in low-quality sites)
    hyphen_count = domain.count('-')
    if hyphen_count > 2:
        score -= 0.1

    # Very short or very long domains
    if len(domain) < 5 or len(domain) > 30:
        score -= 0.05

    return max(0.0, min(1.0, score))


def adjust_credibility_by_context(
    base_score: float,
    has_author: bool = False,
    has_citations: bool = False,
    is_recent: bool = True,
    multiple_sources_confirm: bool = False
) -> float:
    """
    Adjust credibility score based on contextual factors.

    Args:
        base_score: Initial credibility score
        has_author: Whether article has named author
        has_citations: Whether article cites sources
        is_recent: Whether article is recent
        multiple_sources_confirm: Whether claim is confirmed by multiple sources

    Returns:
        Adjusted credibility score
    """
    score = base_score

    if has_author:
        score += 0.05
    if has_citations:
        score += 0.10
    if not is_recent:
        score -= 0.05
    if multiple_sources_confirm:
        score += 0.15

    return max(0.0, min(1.0, score))


def get_credibility_label(score: float) -> str:
    """
    Convert credibility score to human-readable label.

    Args:
        score: Credibility score (0-1)

    Returns:
        Credibility label
    """
    if score >= 0.9:
        return "Very High"
    elif score >= 0.75:
        return "High"
    elif score >= 0.6:
        return "Medium-High"
    elif score >= 0.4:
        return "Medium"
    elif score >= 0.25:
        return "Low"
    else:
        return "Very Low"
