from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment_name: str
    azure_openai_api_version: str = "2023-05-15"

    class Config:
        env_file = ".env"


settings = Settings()
