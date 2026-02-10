from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://app:app@db:5432/app"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    upload_dir: str = "/data/uploads"
    output_dir: str = "/data/outputs"
    frontend_dist_dir: str = "/app/frontend_dist"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
