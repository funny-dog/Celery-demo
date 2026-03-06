"""Tests for crypto module."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet


# Generate a real Fernet key for testing
TEST_KEY = Fernet.generate_key().decode()


def _reload_crypto(encryption_key: str = ""):
    """Reload the crypto module with a patched encryption_key setting."""
    with patch("config.settings") as mock_settings:
        mock_settings.encryption_key = encryption_key
        import crypto

        importlib.reload(crypto)
        return crypto


class TestEncryptDecryptValue:
    """Test encrypt_value / decrypt_value round-trip."""

    def test_round_trip(self):
        mod = _reload_crypto(TEST_KEY)
        plain = "super-secret-password"
        encrypted = mod.encrypt_value(plain)
        assert encrypted.startswith("ENC:")
        assert encrypted != plain
        assert mod.decrypt_value(encrypted) == plain

    def test_decrypt_plaintext_passthrough(self):
        """Values without ENC: prefix are returned as-is."""
        mod = _reload_crypto(TEST_KEY)
        assert mod.decrypt_value("not-encrypted") == "not-encrypted"

    def test_decrypt_non_string_passthrough(self):
        """Non-string values are returned as-is."""
        mod = _reload_crypto(TEST_KEY)
        assert mod.decrypt_value(12345) == 12345  # type: ignore[arg-type]

    def test_encrypt_no_key_returns_plaintext(self):
        """Without encryption key, encrypt_value returns plaintext."""
        mod = _reload_crypto("")
        plain = "my-password"
        assert mod.encrypt_value(plain) == plain

    def test_decrypt_wrong_key_raises(self):
        """Decrypting with a different key raises ValueError."""
        mod1 = _reload_crypto(TEST_KEY)
        encrypted = mod1.encrypt_value("secret")

        other_key = Fernet.generate_key().decode()
        mod2 = _reload_crypto(other_key)
        with pytest.raises(ValueError, match="Decryption failed"):
            mod2.decrypt_value(encrypted)


class TestEncryptDecryptConfig:
    """Test encrypt_sensitive_config / decrypt_sensitive_config."""

    def test_round_trip_config(self):
        mod = _reload_crypto(TEST_KEY)
        config = {
            "host": "db.example.com",
            "port": 5432,
            "database": "mydb",
            "username": "admin",
            "password": "s3cret",
            "secret_key": "aws-secret",
            "access_key": "AKIA1234",
        }
        encrypted = mod.encrypt_sensitive_config(config)

        # Non-sensitive keys are unchanged
        assert encrypted["host"] == "db.example.com"
        assert encrypted["port"] == 5432
        assert encrypted["database"] == "mydb"
        assert encrypted["username"] == "admin"

        # Sensitive keys are encrypted
        assert encrypted["password"].startswith("ENC:")
        assert encrypted["secret_key"].startswith("ENC:")
        assert encrypted["access_key"].startswith("ENC:")

        # Round-trip
        decrypted = mod.decrypt_sensitive_config(encrypted)
        assert decrypted == config

    def test_already_encrypted_not_double_encrypted(self):
        mod = _reload_crypto(TEST_KEY)
        config = {"password": "plain"}
        enc1 = mod.encrypt_sensitive_config(config)
        enc2 = mod.encrypt_sensitive_config(enc1)
        # Should not double-encrypt (ENC: prefix check)
        assert enc1["password"] == enc2["password"]

    def test_config_no_key_passthrough(self):
        mod = _reload_crypto("")
        config = {"host": "localhost", "password": "secret"}
        encrypted = mod.encrypt_sensitive_config(config)
        assert encrypted["password"] == "secret"
        assert encrypted["host"] == "localhost"

    def test_original_config_not_mutated(self):
        mod = _reload_crypto(TEST_KEY)
        config = {"password": "secret", "host": "localhost"}
        original_password = config["password"]
        mod.encrypt_sensitive_config(config)
        assert config["password"] == original_password

    def test_all_sensitive_keys_recognized(self):
        mod = _reload_crypto(TEST_KEY)
        config = {
            "password": "p",
            "secret": "s",
            "token": "t",
            "api_key": "a",
            "private_key": "k",
            "secret_key": "sk",
            "access_key": "ak",
        }
        encrypted = mod.encrypt_sensitive_config(config)
        for key in config:
            assert encrypted[key].startswith("ENC:"), f"{key} was not encrypted"

        decrypted = mod.decrypt_sensitive_config(encrypted)
        assert decrypted == config

    def test_empty_config(self):
        mod = _reload_crypto(TEST_KEY)
        assert mod.encrypt_sensitive_config({}) == {}
        assert mod.decrypt_sensitive_config({}) == {}
