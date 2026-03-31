from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    anthropic_base_url: str
    default_model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    tavily_api_key: str


settings = Settings()
