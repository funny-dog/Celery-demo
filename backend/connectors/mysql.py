"""
MySQL Database Connector
"""
from __future__ import annotations

from typing import Any

import pymysql

from .base import BaseConnector


class MySQLConnector(BaseConnector):
    """MySQL database connector."""

    def connect(self) -> pymysql.Connection:
        if self._connection is None or not self._connection.open:
            self._connection = pymysql.connect(
                host=self.config.get("host", "localhost"),
                port=int(self.config.get("port", 3306)),
                user=self.config.get("username", "root"),
                password=self.config.get("password", ""),
                database=self.config.get("database"),
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=10,
            )
        return self._connection

    def disconnect(self) -> None:
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None

    def test_connection(self) -> tuple[bool, str]:
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result and result.get("1") == 1:
                    return True, "Connection successful"
            return False, "Query returned no result"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_tables(self, schema: str | None = None) -> list[str]:
        conn = self.connect()
        with conn.cursor() as cursor:
            query = "SHOW TABLES"
            cursor.execute(query)
            results = cursor.fetchall()
            # pymysql returns column name as first key
            return [list(row.values())[0] for row in results]

    def get_columns(self, table_name: str) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(f"DESCRIBE `{table_name}`")
            results = cursor.fetchall()
            return [
                {
                    "name": row["Field"],
                    "type": row["Type"],
                    "nullable": row["Null"] == "YES",
                }
                for row in results
            ]

    def get_sample_data(
        self, table_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT %s", (limit,))
            return cursor.fetchall()

    def execute_query(
        self, query: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()
