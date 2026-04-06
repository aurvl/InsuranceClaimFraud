from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import PostgresSettings


def get_postgres_connection(
    settings: PostgresSettings | None = None,
    autocommit: bool = False,
):
    postgres_settings = settings or PostgresSettings.from_env()
    connection = psycopg2.connect(postgres_settings.dsn, cursor_factory=RealDictCursor)
    connection.autocommit = autocommit
    return connection


@contextmanager
def postgres_connection_context(
    settings: PostgresSettings | None = None,
    autocommit: bool = False,
) -> Iterator:
    connection = get_postgres_connection(settings=settings, autocommit=autocommit)
    try:
        yield connection
        if not autocommit:
            connection.commit()
    except Exception:
        if not autocommit:
            connection.rollback()
        raise
    finally:
        connection.close()


def test_postgres_connection(settings: PostgresSettings | None = None) -> bool:
    with postgres_connection_context(settings=settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 AS healthcheck;")
            result = cursor.fetchone()
    return bool(result and result["healthcheck"] == 1)
