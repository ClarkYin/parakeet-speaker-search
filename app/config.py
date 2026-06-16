from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    groq_api_key: str
    hf_token: str
    database_url: str
    deepgram_api_key: Optional[str] = None

settings = Settings()
