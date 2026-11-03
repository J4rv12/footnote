"""FastAPI application object and router wiring."""

from fastapi import FastAPI

from app.config import get_settings
from app.index import CORPUS_DIR, INDEX_DIR, load_index
from app.logs import configure_logging
from app.routes import health

# Eager, at import time, on purpose. Uvicorn imports this module to find
# `app`; if a required setting is missing this raises here, the process exits
# non-zero, and Render fails the deploy while the previous version keeps
# serving. Left lazy, the same mistake boots green and 500s on the first
# real question.
settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="Footnote", version="0.1.0")

# Eager for the same reason: a stale index, or a VOYAGE_* setting on Render
# that differs from the build, fails the deploy here instead of quietly
# ranking chunks at random. Read by /ask from request.app.state.index.
app.state.index = load_index(INDEX_DIR, CORPUS_DIR, settings)

app.include_router(health.router)
