"""
Configuration settings for Weather Service
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings loaded from environment."""
    
    # MongoDB
    mongo_uri: str = os.environ.get("MONGO_DB_URI", "mongodb://mongodb:27017")
    mongo_db: str = os.environ.get("MONGO_DB", "arice")
    
    # Service
    service_name: str = "weather-service"
    service_port: int = int(os.environ.get("PORT", 5002))
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
    
    # FuXi-S2S Model Service
    model_service_url: str = os.environ.get("MODEL_SERVICE_URL", "http://fuxis2s_model:8002")


settings = Settings()
