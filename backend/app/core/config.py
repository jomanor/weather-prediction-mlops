"""Single source of truth for configuration.

Everything the service can be tuned with lives here; no ``os.getenv`` anywhere
else in the package.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "weather-api"
    app_version: str = "2.0.0"
    log_level: str = "INFO"

    # MONGO_URL is accepted for backwards compatibility with the old scripts.
    mongo_uri: str = Field(
        "mongodb://localhost:27017",
        validation_alias=AliasChoices("MONGO_URI", "MONGO_URL"),
    )
    mongo_db: str = "weather_db"
    mongo_timeout_ms: int = 5000

    #: Index creation and registry seeding run once per process, asynchronously,
    #: so a cold request does not pay for them. Set false to skip entirely.
    ensure_indexes_on_start: bool = True

    cors_origins: str = "http://localhost:5173"

    aemet_api_key: str | None = None
    aemet_base_url: str = "https://opendata.aemet.es/opendata/api"
    aemet_timeout_seconds: float = 10.0
    aemet_retries: int = 3
    aemet_backoff_seconds: float = 0.5
    aemet_cache_ttl_seconds: int = 3600

    geo_timeout_seconds: float = 8.0

    #: A served prediction whose source observation is older than this is
    #: flagged ``stale`` (row stays visible, never dropped). The default matches
    #: one ingest/rebuild cycle plus jitter.
    prediction_stale_after_hours: float = Field(default=24.0, ge=0)

    #: Optional ``POST /refresh``, off unless the operator sets
    #: ``REFRESH_ENABLED=true``. The endpoint is unauthenticated and writes a
    #: document to ``raw_weather`` (the largest collection on the free tier), so
    #: exposing it is a deliberate choice. The limiter that caps those writes is
    #: per process: its counters reset whenever the free-tier instance restarts
    #: or sleeps, so it bounds a wake's writes, not a billing period's.
    refresh_enabled: bool = False
    refresh_city_cooldown_seconds: int = 900
    refresh_max_per_minute: int = 6
    refresh_max_per_day: int = 48

    @property
    def cors_origins_list(self) -> list[str]:
        """Comma separated ``CORS_ORIGINS`` as a list, blanks dropped."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
