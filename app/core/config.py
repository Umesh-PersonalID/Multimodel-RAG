# Copyright 2024
# Directory: yt-rag/app/core/config.py

"""
Configuration management for the RAG application.
Handles environment variables and application settings.
"""

import os
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    
    # AI Provider Configuration
    ai_provider: Literal["openai", "anthropic"] = Field(default="openai", env="AI_PROVIDER")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_embed_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBED_MODEL")
    openai_chat_model: str = Field(default="gpt-4o", env="OPENAI_CHAT_MODEL")
    
    # Anthropic Configuration
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_chat_model: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_CHAT_MODEL")
    
    # Application Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # RAG Configuration
    default_top_k: int = Field(default=6)
    chunk_size: int = Field(default=400)  # Approximate tokens
    chunk_overlap: int = Field(default=60)  # 15% overlap
    temperature: float = Field(default=0.1)
    embedding_dimensions: int = Field(default=1536)  # text-embedding-3-small dimensions

    # Multimodal storage
    storage_root: str = Field(default="storage", env="STORAGE_ROOT")
    uploaded_docs_dir: str = Field(default="storage/uploaded_docs", env="UPLOADED_DOCS_DIR")
    extracted_images_dir: str = Field(default="storage/extracted_images", env="EXTRACTED_IMAGES_DIR")

    # Upload ingestion limits (prevents runaway OpenAI calls on image-heavy PDFs)
    upload_vision_caption_enabled: bool = Field(default=False, env="UPLOAD_VISION_CAPTION_ENABLED")
    upload_max_vision_captions: int = Field(default=5, env="UPLOAD_MAX_VISION_CAPTIONS")
    upload_max_images_per_document: int = Field(default=10, env="UPLOAD_MAX_IMAGES_PER_DOCUMENT")
    upload_min_image_dimension: int = Field(default=120, env="UPLOAD_MIN_IMAGE_DIMENSION")
    upload_skip_embedded_images_when_page_has_text: bool = Field(
        default=True, env="UPLOAD_SKIP_EMBEDDED_IMAGES_WHEN_PAGE_HAS_TEXT"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
