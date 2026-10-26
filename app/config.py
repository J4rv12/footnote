"""Every environment variable this service reads, typed and in one place.

`os.getenv` anywhere else in `app/` is a review rejection. The point is that
one file tells you everything the service needs to boot.

Each tunable arrives with the unit that first reads it. A setting nothing
reads is untested, and one whose value nobody has measured yet is a guess.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and `.env`.

    Secrets have no defaults: a missing one must stop the process at boot
    rather than surface as a 500 on the first real question.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Render injects PORT and friends; not our business.
        # A rejected value is otherwise quoted in the error, and the error goes
        # to Render's deploy log. A real key with a stray character would be
        # published there in full.
        hide_input_in_errors=True,
    )

    # --- Secrets. No defaults, ever. ---
    # Both providers check the key on the first call. min_length only stops an
    # empty line in .env (`VOYAGE_API_KEY=`), which reads as "", a valid str,
    # from booting a service whose every question then fails with a 401.
    anthropic_api_key: str = Field(min_length=1)
    voyage_api_key: str = Field(min_length=1)

    # --- Operational settings. Not secret, so defaults are safe here. ---
    # Pinned here and never at call sites (CLAUDE.md section 4). Changing
    # either changes every recorded request, so `make eval` fails until
    # `make eval-record` is rerun.
    anthropic_model: str = "claude-haiku-5-5"
    voyage_model: str = "voyage-4"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built once and cached.

    Raises `pydantic.ValidationError` on the first call if a required
    variable is missing or cannot be coerced to its declared type. Call it
    during app startup so that failure happens at boot, not mid-request.

    Tests override configuration by setting environment variables and then
    calling `get_settings.cache_clear()`.
    """
    return Settings()
