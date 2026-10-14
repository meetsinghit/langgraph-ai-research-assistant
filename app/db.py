"""SQLite persistence for the application layer (Phase 4D.1).

Design constraints (deliberately minimal — stdlib ``sqlite3``, no
ORM, no connection pool):

- **One connection per thread**, created lazily: FastAPI request
  handlers, the run worker threads and SSE streaming threads each
  use their own. WAL journaling + a generous busy timeout make the
  cross-connection access safe under our bounded concurrency; every
  writer uses short transactions.
- The LangGraph checkpoint tables live in a SEPARATE database file
  behind the shared ``SqliteSaver`` (which owns its own connection
  with its own internal lock, per the installed implementation).
  Keeping files separate means application and checkpoint writes
  never contend, and a whole-file retention story stays trivial.
- **The database files are sensitive**: they contain research
  questions, answers and discovered source URLs. Checkpoint rows
  additionally contain bounded webpage excerpts and are pruned when
  a run reaches a terminal state. Deleting rows does NOT guarantee
  secure physical erasure of the SQLite/WAL files (documented
  limitation — see docs/SECURITY.md).
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                      TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idempotency_key         TEXT,
    parent_run_id           TEXT,
    question                TEXT NOT NULL,
    conversation_context    TEXT NOT NULL DEFAULT '',
    status                  TEXT NOT NULL,
    answer                  TEXT,
    review_status           TEXT,
    outcome_status          TEXT,
    outcome_warnings        TEXT NOT NULL DEFAULT '[]',
    error_category          TEXT,
    iterations              INTEGER NOT NULL DEFAULT 0,
    tokens_in               INTEGER NOT NULL DEFAULT 0,
    tokens_out              INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd      REAL NOT NULL DEFAULT 0,
    total_execution_seconds REAL NOT NULL DEFAULT 0,
    metrics_json            TEXT,
    research_depth          TEXT NOT NULL DEFAULT 'standard',
    evidence_policy         TEXT NOT NULL DEFAULT 'balanced',
    created_at              TEXT NOT NULL,
    started_at              TEXT,
    finished_at             TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_idempotency
    ON runs(session_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created_at);

CREATE TABLE IF NOT EXISTS run_sources (
    run_id   TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    sid      TEXT NOT NULL,
    url      TEXT NOT NULL,
    host     TEXT,
    evidence TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (run_id, sid)
);

CREATE TABLE IF NOT EXISTS run_events (
    run_id  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq     INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    type    TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class Database:
    """Thread-safe handle to the application SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        self.initialize()

    #: Additive column migrations for databases created before a
    #: field existed (Phase 4E.1 run metadata). CREATE TABLE IF
    #: NOT EXISTS never alters an existing table, so pre-4E.1
    #: databases get the new columns here — existing rows keep the
    #: documented defaults.
    _MIGRATIONS: tuple[tuple[str, str], ...] = (
        (
            "research_depth",
            "ALTER TABLE runs ADD COLUMN research_depth "
            "TEXT NOT NULL DEFAULT 'standard'",
        ),
        (
            "evidence_policy",
            "ALTER TABLE runs ADD COLUMN evidence_policy "
            "TEXT NOT NULL DEFAULT 'balanced'",
        ),
    )

    def initialize(self) -> None:
        """Create the schema once (idempotent, concurrency-safe)."""
        with self._init_lock:
            if self._initialized:
                return
            connection = self._connect()
            connection.executescript(_SCHEMA)
            existing = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(runs)"
                )
            }
            for column, statement in self._MIGRATIONS:
                if column not in existing:
                    connection.execute(statement)
            connection.commit()
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=10.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
            self._local.connection = connection
        return connection

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        """A committed transaction on this thread's connection."""
        connection = self._connect()
        cursor = connection.cursor()
        try:
            yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None
