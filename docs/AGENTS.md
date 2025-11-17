# Agent Documentation

## Table of Contents

1. [Overview](#overview)
2. [Agent 1: Collector & Indexer](#agent-1-collector--indexer)
3. [Agent 2: Classifier](#agent-2-classifier)
4. [Agent 3: Anomaly Detector](#agent-3-anomaly-detector)
5. [Agent 4: Fact-Checker](#agent-4-fact-checker)
6. [Agent 5: Reporter & Alerter](#agent-5-reporter--alerter)
7. [Agent Communication](#agent-communication)
8. [Creating Custom Agents](#creating-custom-agents)

---

## Overview

The Multi-Agent Disinformation Detection System consists of **five specialized agents**, each responsible for a distinct phase of the fact-checking pipeline. Agents are orchestrated via LangGraph and communicate through a shared state object.

### Agent Lifecycle

```
1. INITIALIZE: Agent receives state from orchestrator
2. VALIDATE: Check if required state fields are present
3. PROCESS: Execute core logic
4. UPDATE: Modify state with results
5. TRACE: Log actions to reasoning_trace
6. RETURN: Pass updated state to next agent
```

### Base Agent Class

All agents inherit from `BaseAgent`:

```python
from abc import ABC, abstractmethod
from typing import Dict, List
from detect.types import FactCheckingState
from detect.utils import setup_logger, MetricsCollector

class BaseAgent(ABC):
    """Base class for all agents in the system"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        self.metrics = MetricsCollector()
        self.name = self.__class__.__name__

    @abstractmethod
    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Main processing logic - must be implemented by subclass"""
        pass

    def validate_state(self, state: FactCheckingState, required_fields: List[str]) -> None:
        """Validate that required state fields are present"""
        for field in required_fields:
            if field not in state or state[field] is None:
                raise ValueError(f"Required field '{field}' missing from state")

    def update_trace(self, state: FactCheckingState, message: str) -> None:
        """Add message to reasoning trace"""
        state['reasoning_trace'].append(f"{self.name}: {message}")
        self.logger.info(message)

    def __call__(self, state: FactCheckingState) -> FactCheckingState:
        """Execute agent with pre/post hooks"""
        self.metrics.increment(f"{self.name}.invocations")
        start_time = time.time()

        try:
            # Pre-processing
            self.logger.info(f"Starting {self.name}")
            state['agents_involved'].append(self.name)

            # Main processing
            result = self.process(state)

            # Post-processing
            duration = time.time() - start_time
            self.metrics.observe(f"{self.name}.duration", duration)
            self.logger.info(f"Completed {self.name} in {duration:.2f}s")

            return result

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            self.metrics.increment(f"{self.name}.errors")
            raise
```

---

## Agent 1: Collector & Indexer

### Purpose

Monitors public data sources, collects potentially viral or trending content, normalizes data, and maintains an index of sources.

### Responsibilities

1. **Data Collection**: Scrape Twitter, Reddit, news RSS feeds, etc.
2. **Normalization**: Convert diverse formats into unified structure
3. **Deduplication**: Detect and merge duplicate claims
4. **Virality Detection**: Identify rapidly spreading content
5. **Source Indexing**: Maintain database of sources with metadata

### Implementation

```python
from detect.agents.base import BaseAgent
from detect.sources import TwitterCollector, RedditCollector, RSSCollector
from detect.utils import normalize_text, detect_duplicates

class CollectorAgent(BaseAgent):
    """Agent 1: Collects and indexes content from public sources"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.twitter = TwitterCollector(config['twitter'])
        self.reddit = RedditCollector(config['reddit'])
        self.rss = RSSCollector(config['rss'])
        self.db = SourceDatabase()

    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Main collection logic"""
        claim = state['original_claim']

        # Normalize input
        normalized_claim = normalize_text(claim)
        state['original_claim'] = normalized_claim

        # Check for duplicates in recent history
        duplicates = detect_duplicates(normalized_claim, window_hours=24)
        if duplicates:
            self.update_trace(state, f"Found {len(duplicates)} similar recent claims")
            state['duplicate_of'] = duplicates[0]['claim_id']
            return state

        # Extract metadata from claim context (if provided)
        if 'context' in state and 'source_url' in state['context']:
            source_info = self._analyze_source(state['context']['source_url'])
            state['source_metadata'] = source_info

        # Search for related content across platforms
        related_content = self._collect_related_content(normalized_claim)
        state['related_content'] = related_content

        self.update_trace(
            state,
            f"Collected {len(related_content)} related items from {len(related_content)} sources"
        )

        return state

    def _analyze_source(self, url: str) -> Dict:
        """Extract source metadata"""
        domain = extract_domain(url)

        # Check if source is in our credibility database
        credibility = self.db.get_source_credibility(domain)

        return {
            'domain': domain,
            'credibility_score': credibility,
            'first_seen': self.db.get_first_seen(domain),
            'category': self.db.get_category(domain)
        }

    def _collect_related_content(self, claim: str, limit: int = 50) -> List[Dict]:
        """Collect related content from multiple sources"""
        results = []

        # Twitter
        try:
            tweets = self.twitter.search(claim, max_results=limit)
            results.extend([{
                'platform': 'twitter',
                'text': t.text,
                'author': t.author_id,
                'engagement': t.public_metrics,
                'created_at': t.created_at
            } for t in tweets])
        except Exception as e:
            self.logger.warning(f"Twitter collection failed: {e}")

        # Reddit
        try:
            posts = self.reddit.search(claim, limit=limit)
            results.extend([{
                'platform': 'reddit',
                'text': p.title + ' ' + p.selftext,
                'subreddit': p.subreddit,
                'score': p.score,
                'created_at': p.created_utc
            } for p in posts])
        except Exception as e:
            self.logger.warning(f"Reddit collection failed: {e}")

        return results
```

### Configuration

```yaml
# config/agents/collector.yaml
collector:
  sources:
    - twitter
    - reddit
    - rss_feeds

  twitter:
    max_results: 100
    search_recent_days: 7
    include_retweets: false

  reddit:
    subreddits:
      - news
      - worldnews
      - politics
    time_filter: week
    limit: 100

  rss_feeds:
    - https://feeds.bbci.co.uk/news/rss.xml
    - https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml

  deduplication:
    similarity_threshold: 0.85
    window_hours: 24
```

### Metrics

| Metric | Description |
|--------|-------------|
| `collector.claims_processed` | Total claims collected |
| `collector.duplicates_found` | Duplicate claims detected |
| `collector.sources_indexed` | New sources added to index |
| `collector.collection_duration` | Time to collect related content |

---

## Agent 2: Classifier

### Purpose

Categorizes claims by topic, extracts entities, decomposes complex claims into verifiable assertions, and assesses complexity.

### Responsibilities

1. **Named Entity Recognition (NER)**: Extract people, organizations, locations, dates
2. **Topic Classification**: Categorize into politics, health, business, etc.
3. **Claim Decomposition**: Break complex claims into atomic assertions
4. **Complexity Scoring**: Rate claim complexity (1-10)
5. **Urgency Assessment**: Determine priority level

### Implementation

```python
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class ClassifierAgent(BaseAgent):
    """Agent 2: Classifies and decomposes claims"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # NER model
        self.nlp = spacy.load("en_core_web_lg")

        # Topic classifier
        self.topic_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        # Claim decomposer (LLM)
        self.llm = LLMService(provider="claude")

        # Embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Main classification logic"""
        claim = state['original_claim']

        # 1. Named Entity Recognition
        entities = self._extract_entities(claim)
        state['entities'] = entities

        # 2. Topic Classification
        topics = self._classify_topics(claim)
        primary_theme = topics[0]['label']
        state['classification'] = {
            'theme': primary_theme,
            'all_topics': topics
        }

        # 3. Claim Decomposition
        assertions = self._decompose_claim(claim)
        state['decomposed_assertions'] = assertions

        # 4. Complexity & Urgency Scoring
        complexity = self._calculate_complexity(claim, assertions, entities)
        urgency = self._assess_urgency(claim, primary_theme)

        state['classification'].update({
            'complexity': complexity,
            'urgency': urgency
        })

        self.update_trace(
            state,
            f"Classified as {primary_theme}, complexity {complexity}, "
            f"decomposed into {len(assertions)} assertions"
        )

        return state

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)

        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities
            'DATE': [],
            'MONEY': [],
            'PERCENT': []
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        return entities

    def _classify_topics(self, text: str) -> List[Dict]:
        """Classify claim into topics"""
        candidate_labels = [
            "politics",
            "health",
            "business",
            "science",
            "technology",
            "climate",
            "sports",
            "entertainment"
        ]

        result = self.topic_classifier(
            text,
            candidate_labels,
            multi_label=False
        )

        return [
            {'label': label, 'score': score}
            for label, score in zip(result['labels'], result['scores'])
        ]

    def _decompose_claim(self, claim: str) -> List[str]:
        """Decompose complex claim into atomic assertions using LLM"""
        prompt = f"""
        Decompose this claim into individual, verifiable assertions.
        Each assertion should be atomic (test one fact only).

        Claim: {claim}

        Return assertions as a JSON array of strings.
        Example: ["Assertion 1", "Assertion 2", "Assertion 3"]
        """

        response = self.llm.generate(
            prompt,
            temperature=0.0,
            response_format="json"
        )

        assertions = json.loads(response)
        return assertions

    def _calculate_complexity(
        self,
        claim: str,
        assertions: List[str],
        entities: Dict
    ) -> int:
        """Calculate claim complexity score (1-10)"""
        score = 0

        # More assertions = more complex
        score += min(len(assertions) * 2, 4)

        # More entities = more complex
        total_entities = sum(len(e) for e in entities.values())
        score += min(total_entities, 3)

        # Claim length factor
        word_count = len(claim.split())
        if word_count > 50:
            score += 2
        elif word_count > 30:
            score += 1

        # Technical terminology
        if any(term in claim.lower() for term in ['statistical', 'percentage', 'algorithm']):
            score += 1

        return min(score, 10)

    def _assess_urgency(self, claim: str, theme: str) -> int:
        """Assess urgency (1-10) based on content and theme"""
        urgency = 5  # Default

        # High urgency themes
        if theme in ['health', 'politics']:
            urgency += 2

        # Keywords indicating high urgency
        urgent_keywords = ['breaking', 'alert', 'urgent', 'emergency', 'pandemic']
        if any(keyword in claim.lower() for keyword in urgent_keywords):
            urgency += 3

        return min(urgency, 10)
```

### Output Example

```json
{
  "entities": {
    "PERSON": ["Jean Dupont"],
    "ORG": ["TechCorp"],
    "PERCENT": ["150%"]
  },
  "classification": {
    "theme": "business",
    "all_topics": [
      {"label": "business", "score": 0.87},
      {"label": "technology", "score": 0.45}
    ],
    "complexity": 6,
    "urgency": 5
  },
  "decomposed_assertions": [
    "Jean Dupont is CEO of TechCorp",
    "TechCorp announced revenue growth",
    "The revenue growth was 150%",
    "The announcement was made in 2024"
  ]
}
```

---

## Agent 3: Anomaly Detector

### Purpose

Identifies semantic anomalies, inconsistencies, and patterns typical of disinformation before full fact-checking.

### Responsibilities

1. **Coherence Analysis**: Check logical consistency
2. **Pattern Detection**: Identify known disinformation patterns
3. **Tone Analysis**: Detect manipulative language
4. **Contradiction Detection**: Find internal inconsistencies
5. **Suspicion Scoring**: Assign early warning scores

### Implementation

```python
class AnomalyDetectorAgent(BaseAgent):
    """Agent 3: Detects semantic anomalies and suspicious patterns"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.llm = LLMService(provider="claude")
        self.pattern_db = DisinformationPatternDatabase()
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Main anomaly detection logic"""
        assertions = state['decomposed_assertions']
        anomaly_scores = {}

        for assertion in assertions:
            score = self._analyze_assertion(assertion, state)
            anomaly_scores[assertion] = score

        state['anomaly_scores'] = anomaly_scores

        # Calculate overall suspicion
        avg_suspicion = sum(anomaly_scores.values()) / len(anomaly_scores)
        state['overall_suspicion'] = avg_suspicion

        high_suspicion = [a for a, s in anomaly_scores.items() if s > 0.7]
        if high_suspicion:
            self.update_trace(
                state,
                f"High suspicion detected in {len(high_suspicion)} assertions"
            )

        return state

    def _analyze_assertion(self, assertion: str, state: FactCheckingState) -> float:
        """Analyze single assertion for anomalies"""
        scores = []

        # 1. Logical coherence
        coherence_score = self._check_coherence(assertion)
        scores.append(coherence_score)

        # 2. Known disinformation patterns
        pattern_score = self._check_patterns(assertion)
        scores.append(pattern_score)

        # 3. Manipulative language
        manipulation_score = self._check_manipulation(assertion)
        scores.append(manipulation_score)

        # 4. Numerical plausibility
        if any(char.isdigit() for char in assertion):
            plausibility_score = self._check_numerical_plausibility(assertion)
            scores.append(plausibility_score)

        return sum(scores) / len(scores)

    def _check_coherence(self, assertion: str) -> float:
        """Check logical coherence using LLM"""
        prompt = f"""
        Analyze the logical coherence of this assertion.
        Does it contain internal contradictions or logical fallacies?

        Assertion: {assertion}

        Return a score from 0 to 1:
        - 0: Perfectly coherent
        - 1: Highly incoherent or contradictory

        Return only the numeric score.
        """

        response = self.llm.generate(prompt, temperature=0.0)
        return float(response.strip())

    def _check_patterns(self, assertion: str) -> float:
        """Check against known disinformation patterns"""
        patterns = [
            {
                'name': 'false_authority',
                'regex': r'(experts say|studies show|scientists confirm)(?! specific)',
                'score': 0.6
            },
            {
                'name': 'emotional_manipulation',
                'keywords': ['shocking', 'outrageous', 'unbelievable', 'they dont want you to know'],
                'score': 0.7
            },
            {
                'name': 'false_urgency',
                'keywords': ['act now', 'before its too late', 'urgent'],
                'score': 0.5
            },
            {
                'name': 'conspiracy_markers',
                'keywords': ['cover-up', 'hidden agenda', 'mainstream media wont tell you'],
                'score': 0.8
            }
        ]

        max_score = 0
        for pattern in patterns:
            if 'regex' in pattern:
                if re.search(pattern['regex'], assertion, re.IGNORECASE):
                    max_score = max(max_score, pattern['score'])
            elif 'keywords' in pattern:
                if any(kw in assertion.lower() for kw in pattern['keywords']):
                    max_score = max(max_score, pattern['score'])

        return max_score

    def _check_manipulation(self, assertion: str) -> float:
        """Detect manipulative language"""
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(assertion)[0]

        # Extreme sentiment can indicate manipulation
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.9:
            return 0.6
        elif sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.9:
            return 0.4

        # Check for absolute certainty markers
        certainty_markers = [
            'definitely', 'absolutely', 'without doubt',
            'proven fact', '100%', 'undeniable'
        ]
        if any(marker in assertion.lower() for marker in certainty_markers):
            return 0.5

        return 0.0

    def _check_numerical_plausibility(self, assertion: str) -> float:
        """Check if numbers seem plausible"""
        # Extract percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', assertion)

        for pct in percentages:
            value = float(pct)
            # Suspiciously round numbers or extreme values
            if value in [100, 200, 300, 500, 1000]:
                return 0.6
            if value > 500:
                return 0.7

        return 0.0
```

---

## Agent 4: Fact-Checker

### Purpose

Core verification engine that retrieves evidence, cross-references sources, and determines verdict for each assertion.

### Responsibilities

1. **Evidence Retrieval**: Hybrid RAG (BM25 + semantic search)
2. **Source Verification**: Check credibility of sources
3. **Cross-Referencing**: Compare multiple sources
4. **Deepfake Detection**: For media content
5. **Verdict Generation**: SUPPORTED/REFUTED/INSUFFICIENT_INFO

### Implementation

```python
class FactCheckerAgent(BaseAgent):
    """Agent 4: Verifies claims against evidence"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.hybrid_retriever = HybridRetriever(config['retrieval'])
        self.llm = LLMService(provider="claude")
        self.credibility_scorer = SourceCredibilityScorer()
        self.deepfake_detector = MultimodalDeepfakeDetector()

    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Main fact-checking logic"""
        assertions = state['decomposed_assertions']
        complexity = state['classification']['complexity']

        # Choose verification strategy based on complexity
        if complexity > 7:
            # Use graph-based reasoning for complex claims
            verdicts = self._verify_with_graph_reasoning(assertions, state)
        else:
            # Standard RAG verification
            verdicts = self._verify_with_rag(assertions, state)

        state['triplet_verdicts'] = verdicts

        # Collect all evidence
        all_evidence = []
        for verdict in verdicts.values():
            all_evidence.extend(verdict.get('evidence', []))

        state['evidence_retrieved'] = all_evidence

        supported = sum(1 for v in verdicts.values() if v['verdict'] == 'SUPPORTED')
        refuted = sum(1 for v in verdicts.values() if v['verdict'] == 'REFUTED')

        self.update_trace(
            state,
            f"Verified {len(assertions)} assertions: "
            f"{supported} supported, {refuted} refuted"
        )

        return state

    def _verify_with_rag(
        self,
        assertions: List[str],
        state: FactCheckingState
    ) -> Dict[str, Dict]:
        """Standard RAG-based verification"""
        verdicts = {}

        for assertion in assertions:
            # Retrieve evidence
            evidence_docs = self.hybrid_retriever.retrieve(assertion, k=10)

            # Score evidence sources
            scored_evidence = []
            for doc in evidence_docs:
                credibility = self.credibility_scorer.score(doc.metadata['source'])
                scored_evidence.append({
                    'text': doc.page_content,
                    'source': doc.metadata['source'],
                    'url': doc.metadata.get('url', ''),
                    'credibility': credibility,
                    'relevance': doc.metadata.get('score', 0.5)
                })

            # LLM verification with Chain-of-Thought
            verdict = self._verify_assertion_with_llm(assertion, scored_evidence)

            verdicts[assertion] = {
                'verdict': verdict['verdict'],
                'confidence': verdict['confidence'],
                'reasoning': verdict['reasoning'],
                'evidence': scored_evidence[:5]  # Top 5
            }

        return verdicts

    def _verify_assertion_with_llm(
        self,
        assertion: str,
        evidence: List[Dict]
    ) -> Dict:
        """Use LLM to verify assertion against evidence"""
        evidence_text = "\n\n".join([
            f"Source: {e['source']} (credibility: {e['credibility']:.2f})\n{e['text']}"
            for e in evidence[:5]
        ])

        prompt = f"""
        Verify this assertion using the provided evidence.
        Use step-by-step reasoning (Chain-of-Thought).

        Assertion: {assertion}

        Evidence:
        {evidence_text}

        Instructions:
        1. Analyze each piece of evidence
        2. Consider source credibility
        3. Look for consensus or contradictions
        4. Apply logical reasoning
        5. Reach a verdict

        Respond in JSON format:
        {{
          "reasoning": "Your step-by-step analysis",
          "verdict": "SUPPORTED" | "REFUTED" | "INSUFFICIENT_INFO",
          "confidence": 0.0-1.0
        }}
        """

        response = self.llm.generate(
            prompt,
            temperature=0.0,
            response_format="json"
        )

        return json.loads(response)

    def _verify_with_graph_reasoning(
        self,
        assertions: List[str],
        state: FactCheckingState
    ) -> Dict[str, Dict]:
        """Graph-based verification for complex claims"""
        # Build claim graph
        graph_builder = ClaimGraphBuilder()
        triplets = []

        for assertion in assertions:
            assertion_triplets = graph_builder.extract_triplets(assertion)
            triplets.extend(assertion_triplets)

        # Verify each triplet
        graph_checker = GraphGuidedFactChecker(triplets, self.hybrid_retriever)
        return graph_checker.verify_triplets_ordered()
```

---

## Agent 5: Reporter & Alerter

### Purpose

Consolidates results from all agents, generates human-readable reports, and triggers alerts when necessary.

### Responsibilities

1. **Verdict Consolidation**: Merge assertion-level verdicts
2. **Report Generation**: Create structured output
3. **Confidence Calculation**: Compute final confidence scores
4. **Alert Decisions**: Trigger notifications for high-priority cases
5. **Recommendation Generation**: Suggest actions

### Implementation

```python
class ReporterAgent(BaseAgent):
    """Agent 5: Generates reports and alerts"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.alert_service = AlertService(config['alerts'])
        self.llm = LLMService(provider="claude")

    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Main reporting logic"""
        # Consolidate verdicts
        final_verdict = self._consolidate_verdicts(state['triplet_verdicts'])
        state['final_verdict'] = final_verdict['verdict']
        state['confidence'] = final_verdict['confidence']

        # Generate explanation
        explanation = self._generate_explanation(state)
        state['explanation'] = explanation

        # Decision: Should we alert?
        if self._should_alert(state):
            alerts = self._generate_alerts(state)
            state['alerts'] = alerts

            # Send alerts
            for alert in alerts:
                self.alert_service.send(alert)

        # Generate recommendations
        recommendations = self._generate_recommendations(state)
        state['recommendations'] = recommendations

        self.update_trace(
            state,
            f"Final verdict: {state['final_verdict']} "
            f"(confidence: {state['confidence']:.2f})"
        )

        return state

    def _consolidate_verdicts(self, triplet_verdicts: Dict) -> Dict:
        """Consolidate assertion-level verdicts into final verdict"""
        verdicts = [v['verdict'] for v in triplet_verdicts.values()]
        confidences = [v['confidence'] for v in triplet_verdicts.values()]

        # If any assertion is REFUTED with high confidence → claim REFUTED
        refuted_count = sum(1 for v in verdicts if v == 'REFUTED')
        supported_count = sum(1 for v in verdicts if v == 'SUPPORTED')
        insufficient_count = sum(1 for v in verdicts if v == 'INSUFFICIENT_INFO')

        total = len(verdicts)

        if refuted_count > 0:
            final_verdict = 'REFUTED'
            final_confidence = sum(
                c for v, c in zip(verdicts, confidences) if v == 'REFUTED'
            ) / refuted_count
        elif supported_count == total:
            final_verdict = 'SUPPORTED'
            final_confidence = sum(confidences) / total
        else:
            final_verdict = 'INSUFFICIENT_INFO'
            final_confidence = 0.5

        return {
            'verdict': final_verdict,
            'confidence': final_confidence,
            'breakdown': {
                'supported': supported_count,
                'refuted': refuted_count,
                'insufficient': insufficient_count
            }
        }

    def _generate_explanation(self, state: FactCheckingState) -> str:
        """Generate human-readable explanation using LLM"""
        prompt = f"""
        Generate a clear, concise explanation of this fact-check result.

        Original Claim: {state['original_claim']}
        Final Verdict: {state['final_verdict']}
        Confidence: {state['confidence']:.2f}

        Assertions and Verdicts:
        {json.dumps(state['triplet_verdicts'], indent=2)}

        Top Evidence:
        {json.dumps(state['evidence_retrieved'][:3], indent=2)}

        Write a 2-3 paragraph explanation suitable for a general audience.
        """

        return self.llm.generate(prompt, temperature=0.3)

    def _should_alert(self, state: FactCheckingState) -> bool:
        """Determine if alert should be sent"""
        # Alert conditions
        conditions = [
            # High-confidence refutation
            state['final_verdict'] == 'REFUTED' and state['confidence'] > 0.8,

            # High urgency theme
            state['classification']['urgency'] >= 8,

            # Viral content being debunked
            len(state.get('related_content', [])) > 100
        ]

        return any(conditions)

    def _generate_alerts(self, state: FactCheckingState) -> List[Dict]:
        """Generate alert payloads"""
        severity = self._calculate_severity(state)

        alerts = []

        # Main alert
        alerts.append({
            'type': 'fact_check_alert',
            'severity': severity,
            'claim_id': state.get('claim_id'),
            'verdict': state['final_verdict'],
            'confidence': state['confidence'],
            'claim': state['original_claim'],
            'explanation': state.get('explanation', ''),
            'evidence_count': len(state['evidence_retrieved']),
            'timestamp': datetime.now().isoformat()
        })

        return alerts

    def _calculate_severity(self, state: FactCheckingState) -> str:
        """Calculate alert severity"""
        if state['confidence'] > 0.9 and state['classification']['urgency'] >= 8:
            return 'CRITICAL'
        elif state['confidence'] > 0.8 and state['classification']['urgency'] >= 6:
            return 'HIGH'
        elif state['confidence'] > 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _generate_recommendations(self, state: FactCheckingState) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if state['final_verdict'] == 'REFUTED':
            recommendations.append(
                "Consider flagging this content for fact-check labels on social platforms"
            )
            recommendations.append(
                "Notify original publisher for correction opportunity"
            )

        if state['confidence'] < 0.7:
            recommendations.append(
                "Manual review recommended due to low confidence"
            )

        if state['classification']['theme'] == 'health':
            recommendations.append(
                "Escalate to health misinformation task force"
            )

        return recommendations
```

---

## Agent Communication

### State Object Structure

```python
class FactCheckingState(TypedDict):
    # Input
    original_claim: str
    context: Optional[Dict]

    # Collector output
    source_metadata: Optional[Dict]
    related_content: List[Dict]
    duplicate_of: Optional[str]

    # Classifier output
    entities: Dict[str, List[Dict]]
    classification: Dict
    decomposed_assertions: List[str]

    # Anomaly Detector output
    anomaly_scores: Dict[str, float]
    overall_suspicion: float

    # Fact-Checker output
    evidence_retrieved: List[Dict]
    triplet_verdicts: Dict[str, Dict]

    # Reporter output
    final_verdict: str
    confidence: float
    explanation: str
    alerts: List[Dict]
    recommendations: List[str]

    # Metadata
    claim_id: Optional[str]
    created_at: str
    agents_involved: List[str]
    reasoning_trace: List[str]
```

---

## Creating Custom Agents

### Step 1: Define Agent Class

```python
from detect.agents.base import BaseAgent
from detect.types import FactCheckingState

class MyCustomAgent(BaseAgent):
    """Custom agent for specific task"""

    def __init__(self, config: Dict):
        super().__init__(config)
        # Initialize custom resources
        self.my_tool = MyTool(config['tool_settings'])

    def process(self, state: FactCheckingState) -> FactCheckingState:
        """Implement custom logic"""
        # Validate required state
        self.validate_state(state, ['original_claim'])

        # Your processing logic
        result = self.my_tool.process(state['original_claim'])

        # Update state
        state['custom_field'] = result

        # Log to trace
        self.update_trace(state, f"Processed with custom logic: {result}")

        return state
```

### Step 2: Register in Workflow

```python
from langgraph.graph import StateGraph

workflow = StateGraph(FactCheckingState)

# Add your custom agent
workflow.add_node("my_custom_agent", MyCustomAgent(config))

# Add edges
workflow.add_edge("classifier", "my_custom_agent")
workflow.add_edge("my_custom_agent", "fact_checker")
```

### Step 3: Configure

```yaml
# config/agents/custom.yaml
my_custom_agent:
  enabled: true
  tool_settings:
    param1: value1
    param2: value2
```

---

## Conclusion

The five agents work in concert to provide comprehensive fact-checking:

1. **Collector**: Gathers and indexes content
2. **Classifier**: Organizes and decomposes claims
3. **Anomaly Detector**: Early warning system
4. **Fact-Checker**: Core verification engine
5. **Reporter**: Consolidates and communicates results

Each agent is modular, testable, and can be enhanced or replaced independently.

For implementation examples, see:
- `detect/agents/` directory
- `tests/agents/` for test cases
- `examples/custom_agent.py` for custom agent template

---

**Last Updated**: 2025-01-15
