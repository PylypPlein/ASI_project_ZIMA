from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = (
        "tmp_kedro/satisfaction-prediction/data/06_models/ag_production.pkl"
    )
    DATABASE_URL: str = "sqlite:///data/predictions.db"
    WANDB_API_KEY: str | None = None

    class ConfigDict:
        env_file = ".env"


settings = Settings()
