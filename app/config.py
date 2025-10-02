"""
Configuration settings for the Sales Analytics Platform
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "sqlite:///./sales_analytics.db"
    database_url_async: str = "sqlite+aiosqlite:///./sales_analytics.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # API
    api_title: str = "Sales Analytics API"
    api_version: str = "1.0.0"
    api_description: str = "Advanced sales analytics and insights platform"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # File Upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: list = [".xlsx", ".xls", ".csv"]
    
    # Analytics
    rfm_segments: int = 5
    forecast_periods: int = 30
    min_purchase_frequency: int = 2
    
    # Caching
    cache_ttl: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
