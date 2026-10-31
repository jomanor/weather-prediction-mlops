"""Single source of truth for configuration.

Everything the service can be tuned with lives here; no ``os.getenv`` anywhere
else in the package.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
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

    #: Shared secret for the registry writes (``POST /cities``,
    #: ``DELETE /cities/{name}``). The registry drives ingestion and the quality
    #: meter, so an anonymous caller must not be able to add or delete a station:
    #: unset (the default) answers both writes with a 503, and an empty
    #: ``CITY_WRITES_TOKEN`` means unset, never "no auth". Reads are unaffected.
    city_writes_token: str | None = None

    #: A served prediction whose source observation is older than
    #: ``PREDICTION_STALE_AFTER_HOURS`` is flagged ``stale`` (row stays visible,
    #: never dropped). The threshold is NOT a setting: it lives in
    #: ``app.core.thresholds`` so the API, the watchdog and the inference job
    #: cannot disagree (audit finding 5).

    @field_validator("city_writes_token", mode="before")
    @classmethod
    def _blank_token_is_unset(cls, value: object) -> object:
        """``CITY_WRITES_TOKEN=`` in a deploy env disables writes, it does not open them."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        """Comma separated ``CORS_ORIGINS`` as a list, blanks dropped."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
