"""
Base Database Connector Interface

Abstract base class for database connectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    """Abstract base class for database connectors."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize connector with configuration.

        Args:
            config: Connection configuration
        """
        self.config = config
        self._connection = None

    @abstractmethod
    def connect(self) -> Any:
        """Establish connection to database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """
        Test if connection is successful.

        Returns:
            (success, message) tuple
        """
        pass

    @abstractmethod
    def get_tables(self, schema: str | None = None) -> list[str]:
        """Get list of table names."""
        pass

    @abstractmethod
    def get_columns(self, table_name: str) -> list[dict[str, Any]]:
        """
        Get column information for a table.

        Returns:
            List of {"name": str, "type": str, "nullable": bool}
        """
        pass

    @abstractmethod
    def get_sample_data(
        self, table_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get sample data from a table."""
        pass

    @abstractmethod
    def execute_query(self, query: str, params: tuple | None = None) -> list[dict[str, Any]]:
        """Execute a SQL query and return results."""
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
