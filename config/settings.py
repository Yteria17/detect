"""Configuration settings for the multi-agent fact-checking system."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "Multi-Agent Fact-Checking System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/factcheck_db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_LLM_MODEL: str = "gpt-4-turbo-preview"

    # Vector DB
    VECTOR_DB_TYPE: str = "chroma"  # chroma, pinecone, weaviate
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None

    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Monitoring
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # Agent Configuration
    FACT_CHECK_TIMEOUT: int = 120  # seconds
    MAX_CONCURRENT_CHECKS: int = 10
    ESCALATION_THRESHOLD: float = 0.6

    # Retrieval
    RETRIEVAL_TOP_K: int = 10
    BM25_WEIGHT: float = 0.5
    SEMANTIC_WEIGHT: float = 0.5

    # Source Credibility Weights
    SOURCE_CREDIBILITY_WEIGHTS: dict = {
        "bbc.com": 0.95,
        "reuters.com": 0.95,
        "apnews.com": 0.95,
        "nature.com": 0.98,
        "sciencedirect.com": 0.93,
        "wikipedia.org": 0.75,
    }

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
