"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
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
