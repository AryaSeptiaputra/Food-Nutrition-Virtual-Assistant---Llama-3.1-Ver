from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    """Application configuration management with environment variable support.
    
    Centralizes all configuration variables for the AI nutritional assistant system.
    Automatically reads configuration from .env file and validates data types using Pydantic.
    All paths are relative to PROJECT_ROOT and support environment variable overrides.
    
    Configuration categories:
    - Project paths: ROOT directory reference
    - MongoDB: Connection strings and database names for session history
    - ChromaDB: Vector database collection and storage paths
    - Models: LLM and embedding model names/paths
    - Data: Food details and nutrition CSV paths
    - Prompts: System prompt template file path
    - Images: Directories for raw, annotated, temporary, and cached images
    
    Usage:
        from src.config.settings import settings
        db_path = settings.CHROMA_DB_PATH
    
    Environment override:
        Can be overridden via .env file variables matching the attribute names.
        Non-matching environment variables are ignored (extra='ignore').
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')

    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    MONGO_CONNECTION_STRING: str
    HISTORY_DATABASE_STRING: str
    HISTORY_COLLECTION_STRING: str

    CHROMA_COLLECTION_NAME: str
    CHROMA_DB_PATH: str

    LLM_MODEL_NAME: str
    KNOWLEDGE_BASE_JSON_SOURCE: str
    EMBEDDING_TEXT_MODEL_NAME: str
    YOLO_MODEL_PATH: str

    FOOD_DETAILS_PATH: str
    FOOD_NUTRITIONS_PATH: str
    TEMPLATE_PROMPT_PATH: str

    RAW_DIR: str
    ANNOTATED_DIR: str
    TEMP_DIR: str
    CACHE_DIR: str

settings = Settings()