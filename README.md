# Système Multi-Agents de Détection de Désinformation

## 🎯 Phase 2 - Implémentation Complète

Système intelligent d'orchestration multi-agents pour détecter, analyser et lutter contre la désinformation sur les réseaux sociaux et sources publiques.

### ✨ Nouveautés Phase 2

- ✅ **Agent 3 - Détecteur d'Anomalies Sémantiques** avec analyse LLM
- ✅ **Orchestration LangGraph** complète des 5 agents
- ✅ **Fact-Checking Avancé** avec RAG hybride (BM25 + Semantic)
- ✅ **Détection de Deepfakes** multimodale (audio + vidéo)
- ✅ **Résolution de preuves contradictoires** par scoring de crédibilité
- ✅ **Graph-based verification** pour claims complexes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│         WORKFLOW MULTI-AGENTS FACT-CHECKING         │
└─────────────────────────────────────────────────────┘

                    [INPUT: Claim]
                           ↓
                  ┌────────────────┐
                  │   Classifier   │  Agent 1
                  │  (Décompose &  │
                  │   Classifie)   │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │   Retriever    │  Agent 2
                  │ (RAG Hybride)  │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │    Anomaly     │  Agent 3 ⭐ NOUVEAU
                  │   Detector     │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │  Fact Checker  │  Agent 4
                  │ (CoT + Graph)  │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │    Reporter    │  Agent 5
                  │  (Génère le    │
                  │   rapport)     │
                  └────────┬───────┘
                           ↓
                   [OUTPUT: Report]
```

---

## 🚀 Installation

### Prérequis
- Python 3.9+
- pip

### Installation des dépendances

```bash
# Installation standard
pip install -r requirements.txt

# Installation avec GPU (optionnel, pour deepfake detection)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configuration

1. Copier le fichier de configuration:
```bash
cp config/config.yaml config/config.local.yaml
```

2. Éditer `config/config.local.yaml` avec vos clés API:
```yaml
llm:
  api_key_env: "ANTHROPIC_API_KEY"  # ou OPENAI_API_KEY
```

3. Définir les variables d'environnement:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

---

## 📖 Utilisation

### Exemple Simple

```python
from workflow import MultiAgentFactChecker

# Initialisation
fact_checker = MultiAgentFactChecker()

# Vérification d'une affirmation
claim = "Le COVID-19 a été créé en laboratoire en 2019."
result = fact_checker.check_claim(claim)

# Résultat
print(f"Verdict: {result['verdict']['verdict_label']}")
print(f"Confiance: {result['verdict']['confidence']:.1%}")
```

### Script de Démonstration

```bash
python example.py
```

Le script propose 5 exemples interactifs:
1. Vérification simple d'une affirmation
2. Vérifications multiples
3. Détection de deepfakes
4. Visualisation du workflow
5. Analyse détaillée complète

---

## 🧩 Composants

### Agents Spécialisés

#### 1. Classifier Agent
- Décomposition en assertions atomiques
- Classification thématique
- Évaluation de complexité et urgence
- Extraction d'entités nommées

#### 2. Retriever Agent
- **RAG Hybride**: BM25 + Semantic Search
- Scoring de crédibilité des sources
- Recherche dans bases de fact-checking
- Web search dynamique (fallback)

#### 3. Anomaly Detector Agent ⭐ NOUVEAU
- Analyse de cohérence logique via LLM
- Détection de patterns manipulateurs
- Analyse linguistique (drapeaux rouges)
- Escalade automatique si suspicion élevée

#### 4. Fact Checker Agent
- **Chain-of-Thought reasoning**
- **Graph-based verification** pour claims complexes
- Résolution de preuves contradictoires
- Scoring de confiance pondéré

#### 5. Reporter Agent
- Consolidation des décisions
- Génération de rapports structurés
- Décision d'alertes intelligentes
- Export JSON/Markdown

### Utilitaires

#### Deepfake Detector 🎥
- Détection audio (CNN + LSTM)
- Détection vidéo (PPG - biological signals)
- Analyse lip-sync
- Fusion multimodale

---

## 📊 Métriques de Performance

| Métrique | Cible Phase 2 | Status |
|----------|---------------|--------|
| Agents implémentés | 5/5 | ✅ |
| Orchestration LangGraph | ✅ | ✅ |
| RAG Hybride | ✅ | ✅ |
| Deepfake Detection | ✅ | ✅ |
| Graph-based Verification | ✅ | ✅ |
| Conflict Resolution | ✅ | ✅ |

---

## 🗂️ Structure du Projet

```
detect/
├── agents/                    # Agents spécialisés
│   ├── __init__.py
│   ├── classifier.py          # Agent 1
│   ├── retriever.py           # Agent 2 (RAG hybride)
│   ├── anomaly_detector.py    # Agent 3 ⭐ NOUVEAU
│   ├── fact_checker.py        # Agent 4 (CoT + Graph)
│   └── reporter.py            # Agent 5
│
├── utils/                     # Utilitaires
│   ├── __init__.py
│   └── deepfake_detector.py   # Détection deepfakes ⭐
│
├── config/                    # Configuration
│   └── config.yaml            # Config par défaut
│
├── data/                      # Données
│   ├── datasets/              # Datasets publics
│   └── models/                # Modèles pré-entraînés
│
├── tests/                     # Tests unitaires
│
├── output/                    # Rapports générés
│   └── reports/
│
├── workflow.py                # Orchestration LangGraph ⭐
├── example.py                 # Script de démonstration
├── requirements.txt           # Dépendances
├── __init__.py
└── README.md                  # Ce fichier
```

---

## 🔬 Exemples d'Utilisation Avancée

### Vérification avec Graph-Based Reasoning

```python
# Pour claims complexes avec multiples assertions liées
complex_claim = """
L'acteur principal de Blade Runner, qui a aussi joué dans Matrix,
a déclaré en 2020 que les voitures volantes seront disponibles en 2025.
"""

result = fact_checker.check_claim(complex_claim)

# Le système utilise automatiquement Graph-based verification
# si complexité > 7
```

### Détection de Deepfake

```python
from utils import DeepfakeDetector

detector = DeepfakeDetector()

# Analyse multimodale (audio + vidéo + lip-sync)
result = detector.detect_multimodal_inconsistency("video.mp4")

print(f"Deepfake: {result['is_deepfake']}")
print(f"Score: {result['deepfake_score']:.1%}")
print(f"Verdict: {result['verdict']}")
```

### Export de Rapports

```python
# Export JSON
fact_checker.export_report(result, "report.json", format='json')

# Export Markdown
fact_checker.export_report(result, "report.md", format='markdown')
```

---

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest

# Avec couverture
pytest --cov=. --cov-report=html

# Tests spécifiques
pytest tests/test_agents.py
pytest tests/test_workflow.py
```

---

## 📚 Documentation Complémentaire

- **[Projet Complet](./projet-multi-agents-desinformation.md)**: Vue d'ensemble, architecture, plan d'implémentation
- **[Détails Techniques](./technique-approfondi.md)**: Implémentation détaillée, code, patterns

---

## 🛣️ Roadmap

### ✅ Phase 1 (Terminée)
- Setup infrastructure
- Agents 1 & 2
- Agent 4 lite
- Dashboard MVP

### ✅ Phase 2 (Terminée - Actuelle)
- ✅ Agent 3 - Détecteur d'Anomalies
- ✅ Orchestration LangGraph
- ✅ Fact-Checking avancé (RAG hybride)
- ✅ Détection deepfakes

### 🔜 Phase 3 (Prochaine)
- API REST FastAPI
- Monitoring & Observabilité
- Tests & Benchmarking
- Déploiement production
- Dashboard Streamlit interactif

---

## 🤝 Contribution

Ce projet est un projet académique de Master IA.

### Développement

```bash
# Formater le code
black .

# Linter
flake8 .

# Type checking
mypy .
```

---

## 📄 Licence

Projet académique - Master IA 2025

---

## 👥 Auteurs

Équipe Detect - Master IA

---

## 🙏 Remerciements

- **LangChain & LangGraph** pour l'orchestration multi-agents
- **Anthropic** pour Claude (LLM)
- Communauté open-source fact-checking
- Datasets publics (Kaggle, data.gouv.fr)

---

## 📞 Support

Pour questions ou problèmes:
1. Voir la documentation dans `/docs`
2. Consulter les exemples dans `example.py`
3. Lire `technique-approfondi.md` pour détails d'implémentation

---

**Version**: 0.2.0 (Phase 2)
**Dernière mise à jour**: 2025-11-17
