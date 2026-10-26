"""Boot-time checks on the secrets.

Each test builds a fresh `Settings` rather than going through `get_settings`,
whose cached instance the rest of the suite shares. `_env_file=None` keeps a
developer's real `.env` out of it; the environment conftest.py set up supplies
every other required value.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def build(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY"])
def test_empty_secret_fails_at_boot(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """An empty line in .env or Render reads as "", which is a valid str."""
    with pytest.raises(ValidationError):
        build(monkeypatch, **{name: ""})


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY"])
def test_missing_secret_fails_at_boot(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """A key forgotten in Render's dashboard must fail the deploy, not the first question."""
    monkeypatch.delenv(name)

    with pytest.raises(ValidationError) as excinfo:
        build(monkeypatch)

    assert name.lower() in str(excinfo.value)
