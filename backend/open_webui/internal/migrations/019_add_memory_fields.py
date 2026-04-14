"""Peewee migrations -- 019_add_memory_fields.py

Adds new columns to the memory table (scope, source_date, importance_score,
access_count, last_accessed_at) and creates the memory_profile table.

Works with both SQLite and PostgreSQL.
"""

from contextlib import suppress

import peewee as pw
from peewee_migrate import Migrator

with suppress(ImportError):
    import playhouse.postgres_ext as pw_pext


def _get_existing_columns(database: pw.Database, table: str) -> set:
    """Return the set of existing column names for a table."""
    if isinstance(database, pw.SqliteDatabase):
        rows = database.execute_sql(f'PRAGMA table_info({table})')
        return {row[1] for row in rows}
    else:
        # PostgreSQL / generic
        rows = database.execute_sql(
            'SELECT column_name FROM information_schema.columns WHERE table_name = %s',
            (table,),
        )
        return {row[0] for row in rows}


def _table_exists(database: pw.Database, table: str) -> bool:
    if isinstance(database, pw.SqliteDatabase):
        rows = database.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return rows.fetchone() is not None
    else:
        rows = database.execute_sql(
            'SELECT to_regclass(%s)', (table,)
        )
        result = rows.fetchone()
        return result is not None and result[0] is not None


def migrate(migrator: Migrator, database: pw.Database, *, fake=False):
    existing_cols = _get_existing_columns(database, 'memory')

    if isinstance(database, pw.SqliteDatabase):
        if 'scope' not in existing_cols:
            database.execute_sql("ALTER TABLE memory ADD COLUMN scope VARCHAR(50) DEFAULT 'general'")
        if 'source_date' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN source_date BIGINT')
        if 'importance_score' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN importance_score REAL DEFAULT 0.5')
        if 'access_count' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN access_count INTEGER DEFAULT 0')
        if 'last_accessed_at' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN last_accessed_at BIGINT')

        database.execute_sql("""
            CREATE TABLE IF NOT EXISTS memory_profile (
                user_id VARCHAR(255) PRIMARY KEY,
                content TEXT NOT NULL,
                fact_count_at_generation INTEGER DEFAULT 0,
                updated_at BIGINT NOT NULL
            )
        """)
    else:
        # PostgreSQL
        if 'scope' not in existing_cols:
            database.execute_sql("ALTER TABLE memory ADD COLUMN scope VARCHAR(50) DEFAULT 'general'")
        if 'source_date' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN source_date BIGINT')
        if 'importance_score' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN importance_score FLOAT DEFAULT 0.5')
        if 'access_count' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN access_count INTEGER DEFAULT 0')
        if 'last_accessed_at' not in existing_cols:
            database.execute_sql('ALTER TABLE memory ADD COLUMN last_accessed_at BIGINT')

        database.execute_sql("""
            CREATE TABLE IF NOT EXISTS memory_profile (
                user_id VARCHAR(255) PRIMARY KEY,
                content TEXT NOT NULL,
                fact_count_at_generation INTEGER DEFAULT 0,
                updated_at BIGINT NOT NULL
            )
        """)


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Rollback is intentionally left as no-op.

    Removing these columns or the profile table would destroy user data.
    """
    pass
