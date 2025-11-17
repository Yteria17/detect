# Data Directory

This directory contains datasets and models used by the Multi-Agent Fact-Checking System.

## Structure

```
data/
├── datasets/     # Public datasets for training and evaluation
└── models/       # Pre-trained models for NLP and deepfake detection
```

## Datasets

### Required Public Datasets

The system can use the following public datasets:

1. **Kaggle Fake News Datasets**
   - Download from: https://www.kaggle.com/datasets
   - Recommended: "Fake News Classification" (43 MB)
   - Place in: `data/datasets/fake_news/`

2. **Climate Misinformation Dataset (data.gouv.fr)**
   - Download from: https://www.data.gouv.fr/
   - French fact-checked climate claims
   - Place in: `data/datasets/climate_misinfo/`

3. **4TU.ResearchData Datasets**
   - Research-quality labeled datasets
   - Place in: `data/datasets/research/`

### Dataset Format

Datasets should follow this structure:
```
datasets/
├── fake_news/
│   ├── train.csv
│   ├── test.csv
│   └── README.md
├── climate_misinfo/
│   └── claims.json
└── research/
    └── labeled_claims.csv
```

Expected columns in CSV:
- `claim`: The text of the claim
- `label`: SUPPORTED / REFUTED / INSUFFICIENT_INFO
- `source`: Source URL (optional)
- `evidence`: Supporting evidence (optional)

## Models

### Pre-trained Models

Place pre-trained models here:

1. **Sentence Embeddings**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Auto-downloaded by the system
   - Cache location: `data/models/embeddings/`

2. **NER Models**
   - spaCy: `en_core_web_sm`
   - Download: `python -m spacy download en_core_web_sm`

3. **Deepfake Detection Models** (optional)
   - Custom CNN models for audio/video analysis
   - Place in: `data/models/deepfake/`

## Usage

The system automatically looks for datasets in this directory. Configure paths in `config/settings.py`:

```python
DATASETS_DIR = "data/datasets"
MODELS_DIR = "data/models"
```

## .gitignore

This directory is git-ignored to avoid committing large files. Use Git LFS if you need to track models.

## Downloading Datasets

Use the provided script to download recommended datasets:

```bash
# Coming soon
python scripts/download_datasets.py
```

Or manually download from the sources listed above.
