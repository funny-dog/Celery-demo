"""Token blacklist using Redis.

Stores hashed tokens with TTL matching their remaining JWT lifetime,
so entries auto-expire and don't accumulate indefinitely.
"""

from __future__ import annotations

import hashlib
import logging

import redis

from config import settings

logger = logging.getLogger(__name__)

KEY_PREFIX = "token_blacklist:"

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    """Lazy-initialise and return the Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )
    return _redis_client


def _token_key(token: str) -> str:
    """Return the Redis key for a token (hashed, not stored raw)."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return f"{KEY_PREFIX}{token_hash}"


def blacklist_token(token: str, expires_in_seconds: int) -> None:
    """Add a token to the blacklist with a TTL.

    Parameters
    ----------
    token:
        The raw JWT string.
    expires_in_seconds:
        Remaining lifetime of the token in seconds.  The Redis key will
        auto-expire after this period so the blacklist stays lean.
    """
    if expires_in_seconds <= 0:
        return
    try:
        _get_redis().setex(_token_key(token), expires_in_seconds, "1")
    except redis.RedisError:
        logger.warning("Failed to blacklist token in Redis", exc_info=True)


def is_token_blacklisted(token: str) -> bool:
    """Check whether a token has been blacklisted.

    Returns ``False`` (allow) when Redis is unreachable so that a Redis
    outage does not lock every user out.
    """
    try:
        return _get_redis().exists(_token_key(token)) > 0
    except redis.RedisError:
        logger.warning("Failed to check token blacklist in Redis", exc_info=True)
        return False
