"""Append-only policy telemetry with checkpoint-addressable histories."""

import json
import sqlite3
from contextlib import closing, contextmanager
from uuid import uuid4


def _connect(path, mode):
    if mode != "rwc" and not path.is_file():
        raise FileNotFoundError(f"Missing policy telemetry journal: {path}")
    return closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode={mode}", uri=True))


class TelemetryJournal:
    def __init__(self, connection, position):
        self.connection = connection
        self.position = position

    @classmethod
    @contextmanager
    def open(cls, path, position=None, *, writer=True):
        if not writer:
            yield cls(None, position)
            return
        if position is None:
            path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(path, "rw" if position is not None else "rwc") as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = EXTRA")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    previous TEXT REFERENCES records(id),
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )
            if (
                position is not None
                and not connection.execute(
                    "SELECT 1 FROM records WHERE id = ?", (position,)
                ).fetchone()
            ):
                raise ValueError(f"Missing committed telemetry position in {path}")
            yield cls(connection, position)

    def append(self, kind, record):
        if self.connection is None:
            return
        position = uuid4().hex
        self.connection.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?)",
            (
                position,
                self.position,
                kind,
                json.dumps(record, allow_nan=False, separators=(",", ":")),
            ),
        )
        self.position = position

    def commit(self):
        # Publish the journal prefix before the checkpoint that references it.
        if self.connection is not None:
            self.connection.commit()
        return self.position

    @staticmethod
    def read(path, position):
        if position is None:
            return {}
        # Following parent links leaves older snapshots readable after a resumed
        # run adds a different suffix to their history.
        # Allow SQLite to roll back an interrupted writer before reading.
        with _connect(path, "rw") as connection:
            rows = connection.execute(
                """WITH RECURSIVE history AS (
                    SELECT rowid, previous FROM records WHERE id = ?
                    UNION ALL
                    SELECT r.rowid, r.previous
                    FROM records r JOIN history h ON r.id = h.previous
                )
                SELECT kind, payload FROM records
                WHERE rowid IN (SELECT rowid FROM history) ORDER BY rowid""",
                (position,),
            )
            history = {}
            for kind, payload in rows:
                history.setdefault(kind, []).append(json.loads(payload))
        if not history:
            raise ValueError(f"Missing committed telemetry position in {path}")
        return history
