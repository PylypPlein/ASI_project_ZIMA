from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = (
        "/app/data/06_models/autogluon_temp_output"
    )
    DATABASE_URL: str = "postgresql://app:app@db:5432/appdb"
    WANDB_API_KEY: str | None = None

    class ConfigDict:
        env_file = ".env"


settings = Settings()
