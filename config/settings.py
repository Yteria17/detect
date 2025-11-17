"""Configuration settings for the multi-agent fact-checking system."""
"""Application settings and configuration."""

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
    """Application settings loaded from environment variables."""

    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None

    # Social Media APIs
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None

    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "misinformation_detector/1.0"

    # News API
    news_api_key: Optional[str] = None

    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "factcheck_db"
    postgres_user: str = "factcheck_user"
    postgres_password: str = ""

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None

    # Weaviate Configuration
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None

    # Application Configuration
    log_level: str = "INFO"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Detection Thresholds
    escalation_confidence_threshold: float = 0.7
    fact_check_timeout: int = 30
    max_evidence_sources: int = 10
    anomaly_threshold: float = 0.6

    # Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-4"
    llm_temperature: float = 0.0
    max_tokens: int = 4000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
settings = Settings()
