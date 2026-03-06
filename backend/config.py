from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://app:app@db:5432/app"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    redis_url: str = "redis://redis:6379/2"
    upload_dir: str = "/data/uploads"
    output_dir: str = "/data/outputs"
    frontend_dist_dir: str = "/app/frontend_dist"

    # Encryption key for sensitive config values (Fernet key)
    # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: str = ""

    # JWT settings
    jwt_secret_key: str = "change-this-to-a-secure-random-string-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 720  # 12 hours

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
