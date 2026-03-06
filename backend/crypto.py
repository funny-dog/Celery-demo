"""
Encryption utilities for sensitive configuration values.

Uses Fernet (AES-128-CBC under the hood) for symmetric encryption.
The encryption key is read from the ENCRYPTION_KEY environment variable via Settings.

If ENCRYPTION_KEY is not set, encryption is skipped (dev mode) with a warning.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from config import settings

logger = logging.getLogger(__name__)

# Keys in connection_config dicts that contain sensitive data
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "secret",
        "token",
        "api_key",
        "private_key",
        "secret_key",
        "access_key",
    }
)

# Prefix that marks a value as already encrypted
_ENC_PREFIX = "ENC:"

# Build the Fernet instance once at module load
_fernet: Fernet | None = None

if settings.encryption_key:
    try:
        _fernet = Fernet(settings.encryption_key.encode())
    except Exception as exc:
        warnings.warn(
            f"Invalid ENCRYPTION_KEY – encryption disabled: {exc}",
            stacklevel=2,
        )
        logger.error("Invalid ENCRYPTION_KEY, encryption will be disabled: %s", exc)
        _fernet = None
else:
    warnings.warn(
        "ENCRYPTION_KEY is not set – sensitive config values will be stored in plaintext. "
        "Set ENCRYPTION_KEY for production use (generate with: "
        'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")',
        stacklevel=2,
    )
    logger.warning("ENCRYPTION_KEY not set; encryption disabled (dev mode).")


def encrypt_value(plaintext: str) -> str:
    """Encrypt a single string value.

    Returns the ciphertext prefixed with ``ENC:``.
    If encryption is disabled (no key), returns the plaintext unchanged.
    """
    if _fernet is None:
        return plaintext
    token = _fernet.encrypt(plaintext.encode())
    return f"{_ENC_PREFIX}{token.decode()}"


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a single ``ENC:``-prefixed string.

    If the value does not start with ``ENC:`` it is returned as-is (already
    plaintext or never encrypted).  If encryption is disabled, the prefix is
    stripped but decryption is skipped (best-effort).
    """
    if not isinstance(ciphertext, str) or not ciphertext.startswith(_ENC_PREFIX):
        return ciphertext

    raw = ciphertext[len(_ENC_PREFIX) :]

    if _fernet is None:
        logger.warning(
            "Encountered ENC:-prefixed value but no ENCRYPTION_KEY is configured; "
            "returning raw ciphertext."
        )
        return raw

    try:
        return _fernet.decrypt(raw.encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt value – wrong key or corrupted ciphertext.")
        raise ValueError("Decryption failed: invalid token or wrong encryption key")


def encrypt_sensitive_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a **copy** of *config* with sensitive keys encrypted.

    Only string values whose key is in :data:`SENSITIVE_KEYS` and that are not
    already ``ENC:``-prefixed will be encrypted.
    """
    result = config.copy()
    for key in SENSITIVE_KEYS:
        value = result.get(key)
        if isinstance(value, str) and not value.startswith(_ENC_PREFIX):
            result[key] = encrypt_value(value)
    return result


def decrypt_sensitive_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a **copy** of *config* with sensitive keys decrypted."""
    result = config.copy()
    for key in SENSITIVE_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.startswith(_ENC_PREFIX):
            result[key] = decrypt_value(value)
    return result
