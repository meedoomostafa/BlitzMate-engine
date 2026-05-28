"""Server configuration loaded from environment variables."""

import os

DEFAULT_DEPTH: int = int(os.getenv("BLITZMATE_DEFAULT_DEPTH", "5"))
MAX_DEPTH: int = int(os.getenv("BLITZMATE_MAX_DEPTH", "6"))

REQUEST_TIMEOUT_S: int = int(os.getenv("BLITZMATE_TIMEOUT_S", "30"))

CACHE_MAX_SIZE: int = 512
CACHE_TTL_S: int = 300
