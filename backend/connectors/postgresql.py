"""
PostgreSQL Database Connector
"""
from __future__ import annotations

from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from .base import BaseConnector


class PostgreSQLConnector(BaseConnector):
    """PostgreSQL database connector."""

    def connect(self) -> psycopg2.extensions.connection:
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                host=self.config.get("host", "localhost"),
                port=int(self.config.get("port", 5432)),
                user=self.config.get("username", "postgres"),
                password=self.config.get("password", ""),
                database=self.config.get("database"),
                connect_timeout=10,
            )
        return self._connection

    def disconnect(self) -> None:
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None

    def test_connection(self) -> tuple[bool, str]:
        try:
            conn = self.connect()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result and result["?column?"] == 1:
                    return True, "Connection successful"
            return False, "Query returned no result"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_tables(self, schema: str | None = None) -> list[str]:
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
            """
            cursor.execute(query, (schema or "public",))
            results = cursor.fetchall()
            return [row["table_name"] for row in results]

    def get_columns(self, table_name: str) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            cursor.execute(query, (table_name,))
            results = cursor.fetchall()
            return [
                {
                    "name": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                }
                for row in results
            ]

    def get_sample_data(
        self, table_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                f'SELECT * FROM "{table_name}" LIMIT %s', (limit,)
            )
            return cursor.fetchall()

    def execute_query(
        self, query: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()
