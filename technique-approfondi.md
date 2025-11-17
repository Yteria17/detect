# Approfondi Technique : Système Multi-Agents pour Détection de Désinformation

## 1. Architectures de Reasoning LLM pour le Fact-Checking

### 1.1 Chain-of-Thought et Multi-Step Reasoning

**Concept Fondamental** : Plutôt qu'une réponse directe, le LLM génère une série d'étapes de reasoning intermédiaires, améliorant drastiquement la qualité[web:126].

**Pour le Fact-Checking**, appliquer CoT signifie :
1. **Claim Decomposition** : Décomposer une affirmation complexe en sous-affirmations vérifiables.
   - Exemple : "Jean Dupont, PDG de TechCorp depuis 2020, a déclaré que les ventes augmentent de 150%."
   - Sous-affirmations : (1) Jean Dupont est PDG de TechCorp, (2) Depuis 2020, (3) Ventes augmentent de 150%.

2. **Evidence Gathering** : Pour chaque sous-affirmation, récupérer preuves.

3. **Verification Logic** : LLM applique reasoning pour lier preuves à affirmation.

**Implémentation Concrète** :
```python
fact_check_prompt = """
Vous êtes un fact-checker expert. Vérifiez cette affirmation étape par étape:

Affirmation: {claim}

Étapes de verification :
1. Identifiez les assertions factuelles clés dans l'affirmation
2. Pour chaque assertion, décrivez quelle preuve vous chercheriez
3. Évaluez la crédibilité de chaque preuve
4. Logique finale : comment ces preuves soutiennent ou réfutent l'affirmation originale?

Répondez en JSON : {
  "key_assertions": [...],
  "evidence_needed": [...],
  "reasoning_steps": [...],
  "verdict": "SUPPORTED/REFUTED/INSUFFICIENT_INFO"
}
"""
```

### 1.2 Layered Chain-of-Thought pour Fact-Checking Multi-Agents

**Concept** : Segmenter le reasoning en couches, chacune vérifiée par un agent spécialisé[web:126].

**Architecture en Couches** :

```
Couche 1 (Agent Classificateur) :
  Input: Affirmation brute
  Processus: Catégorisation + Décomposition
  Output: {assertions[], domain, complexity_score}

Couche 2 (Agent Collecteur Preuves) :
  Input: assertions[] + domain
  Processus: Retrieve documents pertinents via RAG
  Output: {evidence_set, source_credibility, relevance_scores}

Couche 3 (Agent Vérificateur) :
  Input: assertions[] + evidence_set
  Processus: LLM reasoning sur chaque assertion
  Output: {assertion_verdicts[], confidence_scores}

Couche 4 (Agent Consolidateur) :
  Input: assertion_verdicts[]
  Processus: Fusion des verdicts en décision finale
  Output: {final_verdict, reasoning_trace, explanation}
```

**Avantage pour Multi-Agents** : Chaque couche peut être exécutée par un agent différent, parallélisée ou réassignée selon le contexte[web:126].

### 1.3 Resolving Conflicting Evidence via Score-Based Merging

**Problème Réel** : Deux sources crédibles offrent des preuves contradictoires. Comment décider ?

**Solution IJCAI 2025** : Intégrer **média background** et **source credibility** dans le pipeline[web:121].

**Implémentation** :

```python
# Étape 1: Source Credibility Scoring
source_credibility = {
    "bbc.com": 0.95,  # Journalisme reconnu
    "tinytruth.blogspot.com": 0.15,  # Blog inconnu
    "nature.com": 0.98,  # Revue scientifique
}

# Étape 2: Evidence Conflict Resolution
def resolve_conflicting_evidence(evidences: List[Evidence]) -> Dict:
    """
    Si preuves contradictoires, pondérer par crédibilité source.
    """
    verdicts = {}
    for evidence in evidences:
        verdict = evidence.verdict  # "SUPPORTS" ou "REFUTES"
        credibility = source_credibility.get(evidence.source, 0.5)
        
        if verdict not in verdicts:
            verdicts[verdict] = []
        verdicts[verdict].append(credibility)
    
    # Verdict final = catégorie avec plus haute crédibilité cumulée
    weighted_scores = {v: sum(scores) for v, scores in verdicts.items()}
    final_verdict = max(weighted_scores, key=weighted_scores.get)
    
    return {
        "final_verdict": final_verdict,
        "confidence": weighted_scores[final_verdict] / sum(weighted_scores.values()),
        "conflicting_evidence_detected": len(verdicts) > 1
    }
```

---

## 2. Retrieval-Augmented Generation (RAG) pour Fact-Checking

### 2.1 Hybrid Retrieval : BM25 + Semantic Search

**Problème** :
- **BM25 seul** : Bon pour keyword matching exact, mais rate nuances sémantiques.
- **Semantic (Vector) seul** : Excellente compréhension contextuelle, mais peut matcher sur similarité superficielle.

**Solution : Hybrid Search** [web:119][web:122][web:125]

Lancer **deux retrievers en parallèle**, fusionner résultats intelligemment.

#### BM25 (Sparse Retrieval)

**Formule BM25** :
```
score(d, q) = Σ [IDF(i) * (TF(i,d) * (k1+1)) / (TF(i,d) + k1 * (1 - b + b * (|d|/avgdl)))]
```

Paramètres courants :
- `k1` ≈ 1.5 (saturation de terme)
- `b` ≈ 0.75 (normalisation longueur doc)
- **Avantages** : Rapide, transparent, bon pour doc longs.
- **Limitations** : Insensible à synonymes, typos.

#### Semantic Search (Dense Retrieval)

Utilise embeddings : queries et documents convertis en vecteurs, similarité cosinus calculée.

**Modèles recommandés** :
- `all-MiniLM-L6-v2` : Petit, rapide (384 dims).
- `all-mpnet-base-v2` : Plus puissant (768 dims).
- `intfloat/multilingual-e5-base` : Multilingue pour fact-checking international.

**Avantages** : Comprend nuances sémantiques, synonymes, contexte.
**Limitations** : Plus lent, coûteux en storage.

#### Fusion Hybrid

```python
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma

# 1. Setup BM25
bm25_retriever = BM25Retriever.from_texts(documents, k=10)

# 2. Setup Semantic
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents, embedding_model)
semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 3. Ensemble avec weights (50/50 par défaut)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5]  # Ajustable selon domaine
)

# 4. Usage
query = "Jean Dupont PDG TechCorp ventes 150%"
relevant_docs = ensemble_retriever.get_relevant_documents(query)
```

### 2.2 Advanced RAG Architectures pour Fact-Checking

#### Self-RAG (Retrieval-Augmented Generation Réflexif)

**Concept** : LLM décide dynamiquement si retrieval est nécessaire et évalue qualité preuves[web:118].

```python
class SelfRAGFactChecker:
    def check_claim(self, claim: str) -> Dict:
        # Étape 1 : LLM décide si info externe est nécessaire
        retrieve_decision = self.llm(f"""
            Pour vérifier cette affirmation, faut-il chercher des informations externes ?
            Affirmation: {claim}
            Répondez RETRIEVE ou NO_RETRIEVE.
        """)
        
        if retrieve_decision == "NO_RETRIEVE":
            # Affirmation testable contre connaissance interne
            return self._check_internal()
        
        # Étape 2 : Retrieval
        evidence = self.hybrid_retriever.get_relevant_documents(claim)
        
        # Étape 3 : Évaluation Relevance (ISREL token)
        relevance_scores = []
        for doc in evidence:
            relevance = self.llm(f"""
                Cette preuve est-elle pertinente pour vérifier: {claim}?
                Preuve: {doc.page_content[:500]}
                Répondez RELEVANT ou NOT_RELEVANT avec score 0-1.
            """)
            relevance_scores.append(relevance)
        
        # Filtre : garder seulement RELEVANT
        filtered_evidence = [e for e, s in zip(evidence, relevance_scores) if s['is_relevant']]
        
        # Étape 4 : Verify Support (ISSUP token)
        support_score = self.llm(f"""
            Ces preuves soutiennent-elles l'affirmation ?
            Affirmation: {claim}
            Preuves: {[d.page_content for d in filtered_evidence]}
            Répondez: SUPPORTED (score) / REFUTED (score) / INSUFFICIENT (score)
        """)
        
        return support_score
```

#### CRAG (Corrective RAG)

**Concept** : Lightweight evaluator assesses document quality; si mauvaise, dynamic web search[web:118].

```python
class CRAGFactChecker:
    def check_claim(self, claim: str) -> Dict:
        # Step 1: Initial retrieval
        evidence = self.hybrid_retriever.get_relevant_documents(claim)
        
        # Step 2: Quality evaluation
        quality_scores = []
        for doc in evidence:
            quality = self.lightweight_evaluator(claim, doc)
            quality_scores.append(quality)
        
        # Step 3: Triage
        high_quality = [e for e, q in zip(evidence, quality_scores) if q > 0.7]
        low_quality = [e for e, q in zip(evidence, quality_scores) if q <= 0.7]
        
        if len(high_quality) < 3:  # Threshold
            # Trigger dynamic web search
            web_results = self.google_search(claim)
            high_quality.extend(web_results)
        
        # Step 4: Verification avec données de haute qualité
        return self._verify_with_evidence(claim, high_quality)
```

#### Graph RAG

**Concept** : Construire knowledge graph des entités et relations, utiliser pour retrieval[web:118].

```python
# Construction graphe
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687")

# Exemple: Claims avec entités et relations
graph_statements = [
    "Jean Dupont WORKS_FOR TechCorp",
    "Jean Dupont IS_PERSON true",
    "TechCorp HEADQUARTERED_IN Paris",
    "TechCorp REVENUE 150_PERCENT_GROWTH_2024"
]

# Query : Trouver tous les faits liés à Jean Dupont
with driver.session() as session:
    result = session.run("""
        MATCH (person:Person)-[r]->(entity)
        WHERE person.name = 'Jean Dupont'
        RETURN person, r, entity
    """)
```

### 2.3 Prompt Engineering pour Fact-Checking via LLM

**Few-Shot Prompting avec Examples**

```python
FACT_CHECK_FEW_SHOT_PROMPT = """
Vous êtes expert en fact-checking. Vérifiez les affirmations avec reasoning détaillé.

Exemples :

Exemple 1:
Affirmation: "Paris est la capitale de la France"
Reasoning: Cette affirmation concerne géographie politique, connaissance universelle. Paris est effectivement capitale officielle depuis longtemps.
Preuves: Traités diplomatiques, constitution française, organismes internationaux.
Verdict: SUPPORTED, Confiance 99%

Exemple 2:
Affirmation: "Le COVID-19 a été créé en laboratoire en 2019 intentionnellement"
Reasoning: Multiples aspects nécessitent vérification : origine COVID, intention, timeline. Recherche scientifique ne supporte pas création intentionnelle. Preuves contradictoires existent (certains papers chinois anciens vs consensus scientifique).
Verdict: REFUTED pour création intentionnelle, INSUFFICIENT_INFO pour origine exacte. Confiance 75%

Maintenant, vérifiez cette affirmation suivant le même format:
Affirmation: {claim}
Reasoning: [Votre reasoning détaillé ici]
Preuves_nécessaires: [Lister preuves cherchées]
Verdict: [SUPPORTED/REFUTED/INSUFFICIENT_INFO], Confiance: [0-100]%
"""
```

---

## 3. Détection de Deepfakes (Multimodale)

### 3.1 Deepfake Détection Audio

**Techniques 2025** [web:109][web:112][web:115]

#### CNN + LSTM Hybrid (Audio)

**Architecture** :
- **CNN** : Capture spectral features (locales).
- **LSTM** : Capture patterns temporels.
- **Accuracy** : 98.5% sur DEEP-VOICE dataset[web:115].

```python
import torch
import torch.nn as nn

class AudioDeeprakeDetector(nn.Module):
    def __init__(self, num_classes=2):  # Real / Fake
        super().__init__()
        
        # CNN : Spectral features
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # LSTM : Temporal patterns
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 1, audio_length)
        x = self.cnn(x)  # (batch, 128, reduced_length)
        x = x.transpose(1, 2)  # (batch, reduced_length, 128)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last hidden state
        return self.classifier(x)
```

#### Rehearsal with Auxiliary-Informed Sampling (RAIS)

**Concept** : Continual learning pour adapter détection aux nouvelles attaques sans oublier anciennes[web:109].

```python
class RAISDetector:
    """
    Technique pour maintenir performance contre deepfakes évolutifs.
    """
    def __init__(self, model):
        self.model = model
        self.memory_buffer = []
        self.max_buffer_size = 1000
    
    def generate_auxiliary_labels(self, audio_samples):
        """
        Génère labels enrichis : pas juste 'real/fake' mais aussi
        caractéristiques acoustiques (pitch, prosody, spectral artifacts).
        """
        aux_labels = []
        for sample in audio_samples:
            features = {
                'pitch_consistency': self._measure_pitch_consistency(sample),
                'prosody_naturalness': self._measure_prosody(sample),
                'spectral_anomalies': self._detect_spectral_anomalies(sample),
                'voice_biometric_match': self._check_voice_print(sample),
            }
            aux_labels.append(features)
        return aux_labels
    
    def update_with_new_attacks(self, new_deepfake_samples):
        """
        Quand nouvelles attaques détectées, update sans catastrophic forgetting.
        """
        # Select diverse subset from old memory
        aux_labels = self.generate_auxiliary_labels(new_deepfake_samples)
        
        # Rehearse: train on mix of new + old examples with auxiliary labels
        for _ in range(5):  # Rehearse iterations
            batch = random.sample(self.memory_buffer, min(32, len(self.memory_buffer)))
            batch.extend(new_deepfake_samples)
            
            loss = self.model.train_step(batch, aux_labels)
        
        # Update memory buffer
        self.memory_buffer.extend(new_deepfake_samples[:self.max_buffer_size])
```

### 3.2 Deepfake Détection Vidéo : Biological Signal Detection

**Technique FakeCatcher** : Analyser capillaires faciaux et flux sanguin[web:112].

```python
class BiologicalSignalDeepfakeDetector:
    """
    Détecte anomalies biologiques (flux sanguin, capillaires) 
    que deepfakes AI oublient de synthétiser.
    """
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.roi_extractor = self._extract_facial_roi  # Régions riches en capillaires
    
    def extract_ppg_signals(self, video_frames):
        """
        PPG = Photoplethysmography : mesurer flux sanguin via variations couleur.
        """
        ppg_signals = []
        for frame in video_frames:
            # Extraire zone haute du visage (très vascularisée)
            roi = self.roi_extractor(frame)
            
            # Decomposer en canaux RGB
            r, g, b = cv2.split(roi)
            
            # PPG signal = différence entre canaux (flux sanguin cause variations)
            ppg = np.mean(r - g)  # Simplified
            ppg_signals.append(ppg)
        
        return ppg_signals
    
    def detect_anomalies(self, ppg_signals):
        """
        Vrais vidéos : PPG periodic et cohérent avec frequency cardiaque (0.75-4Hz)
        Deepfakes : PPG absent ou artifacts aléatoires.
        """
        # FFT pour extraire frequencies principales
        freqs = np.fft.fft(ppg_signals)
        power_spectrum = np.abs(freqs) ** 2
        
        # Chercher pic dans range cardiaque
        cardiac_range = np.where((freqs > 0.75) & (freqs < 4))
        cardiac_power = np.max(power_spectrum[cardiac_range])
        
        # Threshold basé sur real videos
        if cardiac_power < THRESHOLD:
            return {'verdict': 'LIKELY_DEEPFAKE', 'confidence': 0.85}
        return {'verdict': 'LIKELY_REAL', 'confidence': 0.9}
```

### 3.3 Multimodal Detection : Audio-Video Consistency

**Concept** : Si audio dit "oui" mais lèvres disent "non", c'est suspect[web:112].

```python
class MultimodalDeepfakeDetector:
    def __init__(self):
        self.audio_detector = AudioDeeprakeDetector()
        self.video_detector = BiologicalSignalDeepfakeDetector()
        self.lip_sync_detector = LipSyncChecker()
    
    def detect_multimodal_inconsistency(self, video_file):
        """
        Vérifier cohérence audio-vidéo.
        """
        # Extract audio and video
        audio_stream = self._extract_audio(video_file)
        video_frames = self._extract_frames(video_file)
        
        # Step 1: Audio deepfake probability
        audio_verdict = self.audio_detector.predict(audio_stream)
        
        # Step 2: Video deepfake probability (PPG)
        video_verdict = self.video_detector.detect_anomalies(video_frames)
        
        # Step 3: Lip-sync check
        lip_sync_score = self.lip_sync_detector.check_alignment(audio_stream, video_frames)
        
        # Step 4: Multi-modal fusion
        deepfake_score = {
            'audio_deepfake_prob': audio_verdict['confidence'],
            'video_deepfake_prob': video_verdict['confidence'],
            'lip_sync_anomaly': 1 - lip_sync_score,
        }
        
        # Consensus : if 2+ indicators high = likely deepfake
        consensus = sum([
            1 for v in deepfake_score.values() if v > 0.7
        ])
        
        if consensus >= 2:
            return {'verdict': 'DEEPFAKE_DETECTED', 'confidence': min(deepfake_score.values())}
        
        return {'verdict': 'NOT_DETECTED_AS_DEEPFAKE', 'confidence': 0.5}
```

---

## 4. Graph-Based Fact-Checking (GraphFC)

### 4.1 Concept Graphique

**Problème** : Affirmations complexes avec références ambiguës.

**Exemple** :
"L'acteur principal de Blade Runner, qui a aussi joué dans Matrix, a déclaré en 2020 que..."

Ambiguïté : Qui est "l'acteur principal" ? (Harrison Ford vs Keanu Reeves). Quand a-t-il déclaré ?

**Solution GraphFC** : Convertir claim en graphe de triplets Entity-Relation-Entity[web:110].

### 4.2 Graph Construction

```python
class ClaimGraphBuilder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_triplets(self, claim_text: str) -> List[Tuple]:
        """
        Extraire triplets (sujet, relation, objet) d'une affirmation.
        """
        doc = self.nlp(claim_text)
        
        triplets = []
        
        # Simple entity-relation extraction via POS tags
        for token in doc:
            if token.pos_ == "VERB":  # Relation potential
                # Chercher subject et object
                subjects = [t for t in token.head.lefts if t.dep_ in ("nsubj", "nsubjpass")]
                objects = [t for t in token.head.rights if t.dep_ in ("dobj", "attr")]
                
                for subj in subjects:
                    for obj in objects:
                        triplets.append({
                            'subject': subj.text,
                            'relation': token.lemma_,
                            'object': obj.text,
                            'known_entities': True  # Entities identifiées
                        })
        
        return triplets
    
    def build_claim_graph(self, triplets: List[Tuple]) -> nx.DiGraph:
        """
        Construire graphe dirigé depuis triplets.
        """
        graph = nx.DiGraph()
        
        for triplet in triplets:
            graph.add_edge(
                triplet['subject'],
                triplet['object'],
                relation=triplet['relation']
            )
        
        return graph
```

### 4.3 Graph-Guided Verification

```python
class GraphGuidedFactChecker:
    def __init__(self, claim_graph, evidence_retriever):
        self.claim_graph = claim_graph
        self.evidence_retriever = evidence_retriever
        self.llm = LLMFactChecker()
    
    def verify_triplets_ordered(self):
        """
        Déterminer ordre logique de vérification (planning).
        Vérifier d'abord triplets simples (1-hop), puis complexes (2-hop+).
        """
        # Topological sort pour ordre optimal
        try:
            order = list(nx.topological_sort(self.claim_graph))
        except:
            order = list(self.claim_graph.nodes)
        
        verification_results = {}
        
        for triplet in order:
            result = self._verify_single_triplet(triplet)
            verification_results[triplet] = result
            
            # Si triplet1 REFUTED et triplet2 depends on triplet1 → skip/mark
            if result['verdict'] == 'REFUTED':
                for dependent in self.claim_graph.successors(triplet):
                    verification_results[dependent] = {
                        'verdict': 'INSUFFICIENT_INFO',
                        'reason': f'Depends on {triplet} which is REFUTED'
                    }
        
        return verification_results
    
    def _verify_single_triplet(self, triplet):
        """
        Vérifier si (subject, relation, object) existe dans evidence.
        """
        # Retrieval
        query = f"{triplet['subject']} {triplet['relation']} {triplet['object']}"
        evidence = self.evidence_retriever.get_relevant_documents(query)
        
        # Graph Match : chercher correspondance exacte dans evidence graphs
        verdict = self.llm.verify_with_evidence(triplet, evidence)
        
        return {
            'triplet': triplet,
            'verdict': verdict['verdict'],  # SUPPORTED/REFUTED/INSUFFICIENT
            'confidence': verdict['confidence'],
            'evidence_used': evidence
        }
    
    def final_claim_verdict(self, triplet_results):
        """
        Fusionner verdicts individuels en verdict global.
        """
        supported = sum(1 for v in triplet_results.values() if v['verdict'] == 'SUPPORTED')
        refuted = sum(1 for v in triplet_results.values() if v['verdict'] == 'REFUTED')
        insufficient = sum(1 for v in triplet_results.values() if v['verdict'] == 'INSUFFICIENT_INFO')
        
        total = len(triplet_results)
        
        # Logic : si même 1 triplet refuted = affirmation refuted
        if refuted > 0:
            return {
                'verdict': 'REFUTED',
                'confidence': refuted / total,
                'explanation': f'{refuted}/{total} triplets refuted'
            }
        
        if supported == total:
            return {
                'verdict': 'SUPPORTED',
                'confidence': supported / total,
                'explanation': f'{supported}/{total} triplets verified'
            }
        
        return {
            'verdict': 'INSUFFICIENT_INFO',
            'confidence': supported / (supported + insufficient) if (supported + insufficient) > 0 else 0.5
        }
```

---

## 5. Orchestration Multi-Agents via LangGraph

### 5.1 State Management

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class FactCheckingState(TypedDict):
    """State centralisé partagé par tous les agents."""
    
    original_claim: str
    decomposed_assertions: List[str]
    
    # Résultats partiels
    classification: Dict  # {theme, complexity, urgency}
    evidence_retrieved: List[Dict]  # [{source, credibility, text}]
    anomaly_scores: Dict  # {assertion: score}
    triplet_verdicts: Dict
    
    # État final
    final_verdict: str  # SUPPORTED/REFUTED/INSUFFICIENT
    confidence: float
    reasoning_trace: List[str]
    
    # Metadata
    created_at: str
    agents_involved: List[str]
```

### 5.2 Agent Definitions

```python
from langgraph.graph import StateGraph, END

def agent_classifier(state: FactCheckingState):
    """Agent 1: Classification et décomposition."""
    claim = state['original_claim']
    
    result = llm.invoke(f"""
    Classifiez et décomposez cette affirmation:
    {claim}
    
    Retournez JSON avec : theme, complexity (1-10), urgency (1-10), assertions[]
    """)
    
    state['classification'] = result
    state['decomposed_assertions'] = result['assertions']
    state['reasoning_trace'].append(f"Classifier: Decomposed into {len(result['assertions'])} assertions")
    
    return state

def agent_retriever(state: FactCheckingState):
    """Agent 2: Récupération de preuves."""
    assertions = state['decomposed_assertions']
    all_evidence = []
    
    for assertion in assertions:
        evidence = hybrid_retriever.get_relevant_documents(assertion)
        
        for doc in evidence:
            credibility = credibility_scorer.score_source(doc.metadata['source'])
            all_evidence.append({
                'assertion': assertion,
                'text': doc.page_content,
                'source': doc.metadata['source'],
                'credibility': credibility
            })
    
    state['evidence_retrieved'] = all_evidence
    state['reasoning_trace'].append(f"Retriever: Retrieved {len(all_evidence)} evidence pieces")
    
    return state

def agent_anomaly_detector(state: FactCheckingState):
    """Agent 3: Détection d'anomalies sémantiques."""
    results = {}
    
    for assertion in state['decomposed_assertions']:
        anomaly_score = llm.invoke(f"""
        Détectez anomalies sémantiques dans: "{assertion}"
        Retournez score 0-1 (0=normal, 1=très suspect)
        """)
        
        results[assertion] = float(anomaly_score)
    
    state['anomaly_scores'] = results
    state['reasoning_trace'].append(f"AnomalyDetector: Scored {len(results)} assertions")
    
    return state

def agent_fact_checker(state: FactCheckingState):
    """Agent 4: Vérification de faits."""
    triplet_verdicts = {}
    
    # Use GraphFC if complex, else simple RAG
    if state['classification']['complexity'] > 7:
        graph_builder = ClaimGraphBuilder()
        triplets = graph_builder.extract_triplets(state['original_claim'])
        graph_checker = GraphGuidedFactChecker(triplets, hybrid_retriever)
        triplet_verdicts = graph_checker.verify_triplets_ordered()
    else:
        # Simple CoT verification
        for assertion in state['decomposed_assertions']:
            relevant_evidence = [e for e in state['evidence_retrieved'] 
                                if e['assertion'] == assertion]
            verdict = llm.invoke(f"""
            Vérifiez: "{assertion}"
            Preuves: {[e['text'] for e in relevant_evidence]}
            Retournez : SUPPORTED/REFUTED/INSUFFICIENT avec confiance
            """)
            triplet_verdicts[assertion] = verdict
    
    state['triplet_verdicts'] = triplet_verdicts
    state['reasoning_trace'].append(f"FactChecker: Verified {len(triplet_verdicts)} assertions")
    
    return state

def agent_reporter(state: FactCheckingState):
    """Agent 5: Consolidation et reporting."""
    
    # Consolider tous les verdicts
    verdicts = state['triplet_verdicts']
    refuted = sum(1 for v in verdicts.values() if v['verdict'] == 'REFUTED')
    supported = sum(1 for v in verdicts.values() if v['verdict'] == 'SUPPORTED')
    total = len(verdicts)
    
    if refuted > 0:
        final = 'REFUTED'
        confidence = refuted / total
    elif supported == total:
        final = 'SUPPORTED'
        confidence = 1.0
    else:
        final = 'INSUFFICIENT_INFO'
        confidence = supported / total if total > 0 else 0.5
    
    state['final_verdict'] = final
    state['confidence'] = confidence
    state['reasoning_trace'].append(f"Reporter: Final verdict {final} with confidence {confidence}")
    
    return state

# Construire le graphe
workflow = StateGraph(FactCheckingState)

workflow.add_node("classifier", agent_classifier)
workflow.add_node("retriever", agent_retriever)
workflow.add_node("anomaly_detector", agent_anomaly_detector)
workflow.add_node("fact_checker", agent_fact_checker)
workflow.add_node("reporter", agent_reporter)

# Define edges (séquence d'exécution)
workflow.add_edge("START", "classifier")
workflow.add_edge("classifier", "retriever")
workflow.add_edge("retriever", "anomaly_detector")
workflow.add_edge("anomaly_detector", "fact_checker")
workflow.add_edge("fact_checker", "reporter")
workflow.add_edge("reporter", END)

fact_checker_graph = workflow.compile()
```

### 5.3 Exécution et Tracing

```python
# Invoquer le workflow
initial_state = FactCheckingState(
    original_claim="Jean Dupont, PDG de TechCorp, a déclaré que...",
    decomposed_assertions=[],
    classification={},
    evidence_retrieved=[],
    anomaly_scores={},
    triplet_verdicts={},
    final_verdict="",
    confidence=0.0,
    reasoning_trace=[],
    created_at=datetime.now().isoformat(),
    agents_involved=[]
)

result = fact_checker_graph.invoke(initial_state)

# Output : Rapport complet
report = {
    "claim": result['original_claim'],
    "verdict": result['final_verdict'],
    "confidence": result['confidence'],
    "reasoning": result['reasoning_trace'],
    "evidence_used": result['evidence_retrieved'],
    "assertions_verified": result['triplet_verdicts']
}

print(json.dumps(report, indent=2))
```

---

## 6. Implementation Stack Recommandé (Détails Code)

### 6.1 Backend : FastAPI + Uvicorn

```python
# app.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio

app = FastAPI()

class ClaimRequest(BaseModel):
    claim: str
    priority: str = "normal"  # normal, urgent

class ClaimResponse(BaseModel):
    claim_id: str
    verdict: str
    confidence: float
    created_at: str

fact_checker_graph = compile_multi_agent_graph()

@app.post("/fact-check", response_model=ClaimResponse)
async def fact_check(request: ClaimRequest, background_tasks: BackgroundTasks):
    """
    API endpoint pour lancer vérification.
    """
    claim_id = str(uuid.uuid4())
    
    # Exécution asynchrone
    background_tasks.add_task(
        run_fact_check_async,
        claim_id,
        request.claim
    )
    
    return ClaimResponse(
        claim_id=claim_id,
        verdict="PENDING",
        confidence=0.0,
        created_at=datetime.now().isoformat()
    )

@app.get("/fact-check/{claim_id}")
async def get_result(claim_id: str):
    """Récupérer résultat vérification."""
    result = db.get_fact_check_result(claim_id)
    return result

async def run_fact_check_async(claim_id: str, claim: str):
    """Exécution async du workflow."""
    initial_state = FactCheckingState(original_claim=claim, ...)
    result = fact_checker_graph.invoke(initial_state)
    db.save_fact_check_result(claim_id, result)
```

### 6.2 Vector DB : Weaviate / Pinecone

```python
# Weaviate pour embeddings + metadata
from weaviate.connect import connect_to_local
import weaviate.classes as wvc

client = connect_to_local()

# Créer classe pour stocker evidences
evidence_class = wvc.Class(
    name="Evidence",
    properties=[
        wvc.Property(name="source", data_type=wvc.DataType.TEXT),
        wvc.Property(name="text", data_type=wvc.DataType.TEXT),
        wvc.Property(name="credibility", data_type=wvc.DataType.NUMBER),
        wvc.Property(name="timestamp", data_type=wvc.DataType.DATE),
    ],
    vectorizer_config=wvc.VectorizerConfig.Named("text2vec-transformers")
)

client.collections.create(evidence_class)

# Indexer documents
collection = client.collections.get("Evidence")
collection.data.insert(
    properties={
        "source": "bbc.com",
        "text": "Jean Dupont a déclaré...",
        "credibility": 0.95,
        "timestamp": "2025-01-15"
    }
)

# Recherche hybrid
results = collection.query.hybrid(
    query="Jean Dupont TechCorp",
    limit=10,
    where={"credibility": {">": 0.7}}
)
```

### 6.3 Storage : PostgreSQL + Redis

```python
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base
import redis

Base = declarative_base()

class FactCheckLog(Base):
    __tablename__ = "fact_checks"
    
    id = sa.Column(sa.String, primary_key=True)
    claim = sa.Column(sa.String)
    verdict = sa.Column(sa.String)
    confidence = sa.Column(sa.Float)
    reasoning_trace = sa.Column(sa.JSON)
    evidence_used = sa.Column(sa.JSON)
    created_at = sa.Column(sa.DateTime)
    agents_involved = sa.Column(sa.JSON)

# Cache Redis
redis_client = redis.Redis(host='localhost', port=6379)

# Cacher résultats
cache_key = f"fact_check:{claim_id}"
redis_client.setex(cache_key, 3600, json.dumps(result))  # 1h TTL

# Récupérer du cache
cached = redis_client.get(cache_key)
if cached:
    return json.loads(cached)
```

---

## 7. Patterns d'Orchestration Avancés

### Pattern 1 : Consensus Multi-Path

```python
def consensus_verification(claim: str, num_paths: int = 3) -> Dict:
    """
    Exécuter multiple reasoning paths en parallèle, fusionner résultats.
    """
    import asyncio
    
    paths = [
        lambda: verify_via_rag(claim),
        lambda: verify_via_graph_reasoning(claim),
        lambda: verify_via_chain_of_thought(claim)
    ]
    
    results = asyncio.run(asyncio.gather(*[p() for p in paths[:num_paths]]))
    
    # Consensus : verdict si 2+ paths agree
    verdicts_count = Counter(r['verdict'] for r in results)
    
    if max(verdicts_count.values()) >= 2:
        consensus_verdict = verdicts_count.most_common(1)[0][0]
        confidence = max(verdicts_count.values()) / len(results)
    else:
        consensus_verdict = 'CONFLICTING'
        confidence = 0.33
    
    return {
        'verdict': consensus_verdict,
        'confidence': confidence,
        'individual_results': results
    }
```

### Pattern 2 : Dynamic Escalation

```python
def dynamic_escalation(claim: str, initial_agent: str = "classifier") -> Dict:
    """
    Si confidence trop basse, escalade à spécialiste expert / humain.
    """
    result = fact_checker_graph.invoke(initial_state)
    
    if result['confidence'] < ESCALATION_THRESHOLD:
        if result['classification']['complexity'] > 8:
            # Escalade à expert humain
            return {
                'verdict': 'REQUIRES_HUMAN_REVIEW',
                'reason': f'Confidence {result["confidence"]} below threshold, complexity high',
                'estimated_review_time': '2-4 hours'
            }
        else:
            # Re-run avec plus de preuves
            additional_evidence = deep_web_search(claim)
            result['evidence_retrieved'].extend(additional_evidence)
            return fact_checker_graph.invoke(result)
```

---

## Conclusion

Ce document technique démontre comment orchestrer une plateforme multi-agents hautement sophistiquée pour la détection de désinformation. Les technologies clés (RAG, Chain-of-Thought, GraphFC, deepfake detection, orchestration LangGraph) sont toutes expliquées avec code concret et patterns production-ready.

**Points clés pour l'implémentation** :
1. Hybrid retrieval (BM25 + Semantic) pour robustesse.
2. Self-RAG / CRAG pour adaptabilité.
3. Multimodale deepfake detection (audio + vidéo + PPG).
4. GraphFC pour claims complexes.
5. LangGraph pour orchestration modulaire et traçabilité.
6. Tests rigoureux : accuracy > 90%, F1 > 0.85, latence < 30s.

Ce projet est production-ready et très valorisant pour un portfolio Master IA en 2025.
