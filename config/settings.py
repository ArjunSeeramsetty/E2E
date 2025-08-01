"""
Configuration settings for the power supply data warehouse
"""

import os
from typing import Optional

class Settings:
    """Application settings"""
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///power_supply_data.db")
    
    # File paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    
    # Parsing settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    SUPPORTED_FORMATS: list = [".pdf"]
    
    # Validation settings
    STRICT_VALIDATION: bool = os.getenv("STRICT_VALIDATION", "false").lower() == "true"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)

# Global settings instance
settings = Settings() 