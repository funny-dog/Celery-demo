"""
Masking Rules Engine

Provides various data masking strategies and a rule engine for applying them.
"""
from __future__ import annotations

import hashlib
import random
import string
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class MaskingType(str, Enum):
    """Types of masking strategies."""
    MASK = "mask"  # Replace with * or other character
    REPLACE = "replace"  # Replace with fixed value
    HASH = "hash"  # SHA256 hash
    ENCRYPT = "encrypt"  # Format-preserving encryption
    REDACT = "redact"  # Complete redaction
    GENERALIZE = "generalize"  # Reduce precision (e.g., age ranges)
    PERTURB = "perturb"  # Add noise to numeric values
    NULLIFY = "nullify"  # Set to NULL/empty


class MaskingStrategy(ABC):
    """Base class for masking strategies."""

    @abstractmethod
    def mask(self, value: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply masking to a value."""
        pass

    @property
    @abstractmethod
    def masking_type(self) -> MaskingType:
        """Return the masking type."""
        pass


class EmailMasking(MaskingStrategy):
    """Mask email addresses: john@example.com -> j***@example.com"""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""
        text = str(value)
        if "@" not in text:
            return _generic_mask(text)

        local, domain = text.split("@", 1)
        if not local:
            return f"***@{domain}"
        if len(local) == 1:
            return f"{local}***@{domain}"
        return f"{local[0]}{'*' * (len(local) - 1)}@{domain}"

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.MASK


class PhoneMasking(MaskingStrategy):
    """Mask phone numbers: 13812345678 -> 138****5678"""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""
        text = str(value)

        # Extract digits
        digits = [c for c in text if c.isdigit()]
        if len(digits) < 4:
            return "*" * len(text)

        # Keep first 3 and last 4
        keep_start = 3
        keep_end = 4

        result = []
        digit_index = 0
        for char in text:
            if char.isdigit():
                if digit_index < keep_start or digit_index >= len(digits) - keep_end:
                    result.append(char)
                else:
                    result.append("*")
                digit_index += 1
            else:
                result.append(char)

        return "".join(result)

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.MASK


class IDCardMasking(MaskingStrategy):
    """Mask ID card: 110101199001011234 -> 11************34"""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""
        text = str(value).strip()

        if len(text) <= 4:
            return "*" * len(text)

        # Keep first 2 and last 2
        return f"{text[:2]}{'*' * (len(text) - 4)}{text[-2:]}"

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.MASK


class NameMasking(MaskingStrategy):
    """Mask names: 张三 -> 张*"""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""
        text = str(value).strip()

        if len(text) <= 1:
            return "*" * len(text)

        # For Chinese names, keep surname only
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            return f"{text[0]}{'*' * (len(text) - 1)}"

        # For Western names, keep first letter of first name
        parts = text.split()
        if len(parts) > 1:
            return f"{parts[0][0]}.{' '.join(parts[1:])}"

        return f"{text[0]}{'*' * (len(text) - 1)}"

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.MASK


class AddressMasking(MaskingStrategy):
    """Mask addresses: keep first 6 chars"""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""
        text = str(value).strip()

        if len(text) <= 6:
            return "*" * len(text)

        return f"{text[:6]}{'*' * (len(text) - 6)}"

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.MASK


class HashMasking(MaskingStrategy):
    """SHA256 hash masking with optional salt."""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""

        params = params or {}
        salt = params.get("salt", "")
        text = f"{value}{salt}"
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.HASH


class FixedReplaceMasking(MaskingStrategy):
    """Replace with a fixed value."""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        params = params or {}
        return params.get("replace_value", "***REDACTED***")

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.REPLACE


class RedactMasking(MaskingStrategy):
    """Complete redaction - replace with asterisks."""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""
        text = str(value)
        return "*" * len(text)

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.REDACT


class GeneralizeAgeMasking(MaskingStrategy):
    """Generalize age to ranges."""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> str:
        if not value:
            return ""

        try:
            age = int(value)
            if age < 18:
                return "<18"
            elif age < 30:
                return "18-29"
            elif age < 45:
                return "30-44"
            elif age < 60:
                return "45-59"
            else:
                return "60+"
        except (ValueError, TypeError):
            return str(value)

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.GENERALIZE


class NullifyMasking(MaskingStrategy):
    """Set value to NULL/empty."""

    def mask(self, value: Any, params: dict[str, Any] | None = None) -> None:
        return None

    @property
    def masking_type(self) -> MaskingType:
        return MaskingType.NULLIFY


# Registry of available masking strategies
MASKING_REGISTRY: dict[str, type[MaskingStrategy]] = {
    "email": EmailMasking,
    "phone": PhoneMasking,
    "id_card": IDCardMasking,
    "name": NameMasking,
    "address": AddressMasking,
    "hash": HashMasking,
    "replace": FixedReplaceMasking,
    "redact": RedactMasking,
    "generalize_age": GeneralizeAgeMasking,
    "nullify": NullifyMasking,
}


def get_masking_strategy(masking_type: str) -> MaskingStrategy:
    """Get a masking strategy by name."""
    if masking_type not in MASKING_REGISTRY:
        raise ValueError(
            f"Unknown masking type: {masking_type}. "
            f"Available: {', '.join(MASKING_REGISTRY.keys())}"
        )
    return MASKING_REGISTRY[masking_type]()


def _generic_mask(value: str, keep_start: int = 1, keep_end: int = 1) -> str:
    """Generic masking helper."""
    if not value:
        return ""
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return f"{value[:keep_start]}{'*' * (len(value) - keep_start - keep_end)}{value[-keep_end:]}"
