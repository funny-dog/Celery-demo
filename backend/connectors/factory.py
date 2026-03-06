"""
Connector Factory

Creates database connectors based on connection type.
"""
from __future__ import annotations

from typing import Any

from .base import BaseConnector


def create_connector(
    connector_type: str, config: dict[str, Any]
) -> BaseConnector:
    """
    Create a database connector based on type.

    Args:
        connector_type: Type of connector (mysql, postgresql, etc.)
        config: Connection configuration

    Returns:
        Database connector instance

    Raises:
        ValueError: If connector type is not supported
    """
    connectors = {
        "mysql": "MySQLConnector",
        "postgresql": "PostgreSQLConnector",
        # Add more as implemented
    }

    if connector_type not in connectors:
        available = ", ".join(connectors.keys())
        raise ValueError(
            f"Unsupported connector type: {connector_type}. "
            f"Available: {available}"
        )

    module_name = f".{connector_type}"
    module = __import__(module_name, fromlist=[connectors[connector_type]], level=1)
    connector_class = getattr(module, connectors[connector_type])

    return connector_class(config)


def get_available_connectors() -> list[str]:
    """Get list of available connector types."""
    return ["mysql", "postgresql"]
