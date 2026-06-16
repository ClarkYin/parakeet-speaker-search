from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    together_api_key: str
    hf_token: str
    database_url: str

settings = Settings()
