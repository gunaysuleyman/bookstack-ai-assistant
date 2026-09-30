import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from adaptive.catalog import CatalogFilter, location_text, real_name
from adaptive.contracts import AuthorizationScope, SyncJob

# bm25 column weights: chunk_id, page_id, revision_id (unindexed), body, title, location.
FTS_WEIGHTS = (0.0, 0.0, 0.0, 1.0, 1.5, 1.0)


def connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


class _Snapshot:
    """Rows copied while the connection lock is held, so another thread cannot invalidate the cursor."""

    def __init__(self, rows, rowcount: int, lastrowid):
        self._rows = rows
        self._index = 0
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self):
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self):
        rows = self._rows[self._index :]
        self._index = len(self._rows)
        return rows


class _LockedConnection:
    """Serializes statements. Multi-statement transactions hold the same lock."""

    def __init__(self, raw: sqlite3.Connection, lock: threading.RLock):
        self._raw = raw
        self._lock = lock

    def execute(self, *args, **kwargs):
        with self._lock:
            cursor = self._raw.execute(*args, **kwargs)
            lastrowid = cursor.lastrowid
            rowcount = cursor.rowcount
            try:
                rows = cursor.fetchall()
            except sqlite3.Error:
                rows = []
            return _Snapshot(rows, rowcount, lastrowid)

    def executemany(self, *args, **kwargs):
        with self._lock:
            cursor = self._raw.executemany(*args, **kwargs)
            return _Snapshot([], cursor.rowcount, cursor.lastrowid)

    def executescript(self, *args, **kwargs):
        with self._lock:
            return self._raw.executescript(*args, **kwargs)

    def close(self):
        with self._lock:
            return self._raw.close()


class StateStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._raw = connect(path)
        self.conn = _LockedConnection(self._raw, self._lock)
        self._init()

    def close(self) -> None:
        self.conn.close()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS scopes (
                scope_ref TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                acl_version TEXT NOT NULL,
                expires_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sync_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER NOT NULL,
                event TEXT NOT NULL,
                generation INTEGER NOT NULL,
                status TEXT NOT NULL,
                lease_owner TEXT,
                lease_until REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 5,
                next_attempt_at REAL NOT NULL,
                last_error TEXT,
                payload_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON sync_jobs(status, next_attempt_at);
            CREATE TABLE IF NOT EXISTS page_state (
                page_id INTEGER PRIMARY KEY,
                source_updated_at TEXT,
                content_hash TEXT,
                metadata_hash TEXT,
                active_revision TEXT,
                chunk_schema_version TEXT,
                embedding_model_id TEXT,
                status TEXT NOT NULL,
                generation INTEGER NOT NULL DEFAULT 0,
                book_id INTEGER,
                book_name TEXT,
                chapter_name TEXT,
                shelf_names TEXT,
                title TEXT,
                url TEXT,
                tags_str TEXT,
                error TEXT,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS revisions (
                revision_id TEXT PRIMARY KEY,
                page_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                generation INTEGER NOT NULL,
                content_hash TEXT,
                metadata_hash TEXT,
                page_json TEXT,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS parent_records (
                parent_id TEXT PRIMARY KEY,
                page_id INTEGER NOT NULL,
                revision_id TEXT NOT NULL,
                heading TEXT,
                body TEXT,
                ordinal INTEGER
            );
            CREATE TABLE IF NOT EXISTS chunk_records (
                chunk_id TEXT PRIMARY KEY,
                page_id INTEGER NOT NULL,
                revision_id TEXT NOT NULL,
                parent_id TEXT,
                heading TEXT,
                body TEXT,
                embed_text TEXT,
                ordinal INTEGER
            );
            CREATE TABLE IF NOT EXISTS catalog_pages (
                page_id INTEGER PRIMARY KEY,
                title TEXT,
                url TEXT,
                book_id INTEGER,
                book_name TEXT,
                chapter_id INTEGER,
                chapter_name TEXT,
                shelf_names TEXT,
                tags_str TEXT,
                revision_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_catalog_book_sort ON catalog_pages(book_name, title);
            CREATE INDEX IF NOT EXISTS idx_catalog_book_id ON catalog_pages(book_id);
            CREATE INDEX IF NOT EXISTS idx_page_state_published ON page_state(status, page_id);
            CREATE TABLE IF NOT EXISTS vector_gc (
                chunk_id TEXT PRIMARY KEY,
                revision_id TEXT
            );
            CREATE TABLE IF NOT EXISTS shadow_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                query_fp TEXT,
                legacy_pages TEXT,
                adaptive_pages TEXT,
                latency_ms REAL
            );
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                purpose TEXT,
                provider TEXT,
                model TEXT,
                prompt_tokens_est INTEGER,
                completion_tokens_est INTEGER,
                prompt_tokens_actual INTEGER,
                completion_tokens_actual INTEGER,
                latency_ms REAL
            );
            CREATE TABLE IF NOT EXISTS assistant_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                user_id TEXT,
                query TEXT,
                answer TEXT,
                provider TEXT,
                model TEXT,
                reasoning_effort TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                input_usd_per_mtok REAL,
                output_usd_per_mtok REAL,
                cost_usd REAL,
                evidence_json TEXT
            );
            """
        )
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(assistant_turns)").fetchall()}
        if "evidence_json" not in columns:
            self.conn.execute("ALTER TABLE assistant_turns ADD COLUMN evidence_json TEXT")
        chunk_columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunk_records)").fetchall()}
        if "fts_rowid" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunk_records ADD COLUMN fts_rowid INTEGER")
        self.conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_chunk_page_rev ON chunk_records(page_id, revision_id);
            CREATE INDEX IF NOT EXISTS idx_chunk_rev ON chunk_records(revision_id);
            CREATE INDEX IF NOT EXISTS idx_parent_page_rev ON parent_records(page_id, revision_id);
            CREATE INDEX IF NOT EXISTS idx_parent_rev ON parent_records(revision_id);
            CREATE INDEX IF NOT EXISTS idx_revisions_page ON revisions(page_id, state);
            CREATE INDEX IF NOT EXISTS idx_catalog_chapter ON catalog_pages(chapter_id);
            """
        )
        fts_columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks_fts)").fetchall()}
        rebuild_fts = bool(fts_columns) and "location" not in fts_columns
        if rebuild_fts:
            self.conn.execute("DROP TABLE chunks_fts")
        # page_id is UNINDEXED, so FTS rows are always deleted by rowid
        # (chunk_records.fts_rowid); a WHERE page_id delete scans the table.
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                page_id UNINDEXED,
                revision_id UNINDEXED,
                body,
                title,
                location,
                tokenize = "unicode61 remove_diacritics 2"
            )
            """
        )
        if rebuild_fts:
            self._rebuild_fts()
        self._readers = threading.local()

    def _rebuild_fts(self) -> None:
        """Refill FTS from chunk_records after a schema change, keeping lexical search usable before a reindex."""
        rows = self.conn.execute(
            """
            SELECT c.rowid AS rid, c.chunk_id, c.page_id, c.revision_id, c.heading, c.body,
                   p.title, p.book_name, p.chapter_name, p.shelf_names
            FROM chunk_records c LEFT JOIN page_state p ON p.page_id = c.page_id
            """
        ).fetchall()
        with self.transaction():
            self.conn.executemany(
                "INSERT INTO chunks_fts(rowid, chunk_id, page_id, revision_id, body, title, location) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["rid"],
                        row["chunk_id"],
                        str(row["page_id"]),
                        row["revision_id"],
                        f"{row['heading'] or ''}\n{row['body'] or ''}",
                        str(row["title"] or ""),
                        location_text(row["book_name"], row["chapter_name"], _shelf_names(row["shelf_names"])),
                    )
                    for row in rows
                ],
            )
            self.conn.execute("UPDATE chunk_records SET fts_rowid = rowid")

    def _reader(self):
        """Per-thread read-only connection, so searches do not wait on the writer lock (WAL readers run concurrently)."""
        conn = getattr(self._readers, "conn", None)
        if conn is None:
            if self.path == ":memory:":
                return self.conn
            conn = sqlite3.connect(self.path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA query_only=1")
            self._readers.conn = conn
        return conn

    def transaction(self):
        return _Transaction(self._raw, self._lock)

    def save_scope(self, scope_ref: str, scope: AuthorizationScope) -> None:
        self.purge_expired_scopes()
        payload = scope.model_dump()
        self.conn.execute(
            """
            INSERT INTO scopes(scope_ref, payload_json, fingerprint, acl_version, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scope_ref) DO UPDATE SET
                payload_json=excluded.payload_json,
                fingerprint=excluded.fingerprint,
                acl_version=excluded.acl_version,
                expires_at=excluded.expires_at
            """,
            (scope_ref, json.dumps(payload), scope.fingerprint, scope.acl_version, scope.expires_at),
        )

    def purge_expired_scopes(self, now: Optional[int] = None) -> int:
        current = int(time.time() if now is None else now)
        removed = self.conn.execute("DELETE FROM scopes WHERE expires_at <= ?", (current,))
        return int(removed.rowcount)

    def get_scope(self, scope_ref: str, now: Optional[int] = None) -> Optional[AuthorizationScope]:
        current = int(time.time() if now is None else now)
        row = self.conn.execute("SELECT payload_json, expires_at FROM scopes WHERE scope_ref = ?", (scope_ref,)).fetchone()
        if row is None:
            return None
        if current >= int(row["expires_at"]):
            self.conn.execute("DELETE FROM scopes WHERE scope_ref = ?", (scope_ref,))
            return None
        return AuthorizationScope.model_validate(json.loads(row["payload_json"]))

    def enqueue(self, page_id: int, event: str, payload: Optional[dict] = None, max_attempts: int = 5) -> int:
        now = time.time()
        with self.transaction():
            current = self.conn.execute("SELECT COALESCE(MAX(generation), 0) AS gen FROM sync_jobs WHERE page_id = ?", (page_id,)).fetchone()
            state = self.conn.execute("SELECT generation FROM page_state WHERE page_id = ?", (page_id,)).fetchone()
            generation = max(int(current["gen"] if current else 0), int(state["generation"] if state else 0)) + 1
            self.conn.execute(
                "UPDATE sync_jobs SET status = 'superseded', updated_at = ? WHERE page_id = ? AND status = 'queued'",
                (now, page_id),
            )
            cursor = self.conn.execute(
                """
                INSERT INTO sync_jobs(
                    page_id, event, generation, status, attempts, max_attempts,
                    next_attempt_at, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?)
                """,
                (page_id, event, generation, max_attempts, now, json.dumps(payload or {}), now, now),
            )
            return int(cursor.lastrowid)

    def release_expired_leases(self, now: Optional[float] = None) -> int:
        current = time.time() if now is None else now
        with self.transaction():
            rows = self.conn.execute(
                "SELECT id, attempts, max_attempts FROM sync_jobs WHERE status = 'leased' AND lease_until IS NOT NULL AND lease_until < ?",
                (current,),
            ).fetchall()
            dead = 0
            for row in rows:
                attempts = int(row["attempts"]) + 1
                if attempts >= int(row["max_attempts"]):
                    self.conn.execute(
                        "UPDATE sync_jobs SET status = 'dead', attempts = ?, last_error = ?, updated_at = ? WHERE id = ?",
                        (attempts, "lease expired", current, row["id"]),
                    )
                    dead += 1
                else:
                    self.conn.execute(
                        """
                        UPDATE sync_jobs
                        SET status = 'queued', attempts = ?, lease_owner = NULL, lease_until = NULL,
                            next_attempt_at = ?, last_error = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (attempts, current, "lease expired", current, row["id"]),
                    )
            return dead

    def lease_next(self, owner: str, lease_seconds: int, now: Optional[float] = None) -> Optional[SyncJob]:
        current = time.time() if now is None else now
        self.release_expired_leases(current)
        with self.transaction():
            row = self.conn.execute(
                """
                SELECT * FROM sync_jobs
                WHERE status = 'queued' AND next_attempt_at <= ?
                ORDER BY generation ASC, id ASC
                LIMIT 1
                """,
                (current,),
            ).fetchone()
            if row is None:
                return None
            updated = self.conn.execute(
                """
                UPDATE sync_jobs
                SET status = 'leased', lease_owner = ?, lease_until = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (owner, current + lease_seconds, current, row["id"]),
            )
            if updated.rowcount != 1:
                return None
        return self._job(row)

    def complete_job(self, job_id: int) -> None:
        now = time.time()
        self.conn.execute(
            "UPDATE sync_jobs SET status = 'done', lease_owner = NULL, lease_until = NULL, updated_at = ? WHERE id = ?",
            (now, job_id),
        )

    def fail_job(self, job_id: int, error: str, backoff_s: float = 2.0) -> None:
        now = time.time()
        row = self.conn.execute("SELECT attempts, max_attempts FROM sync_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return
        attempts = int(row["attempts"]) + 1
        status = "dead" if attempts >= int(row["max_attempts"]) else "queued"
        self.conn.execute(
            """
            UPDATE sync_jobs
            SET status = ?, attempts = ?, last_error = ?, next_attempt_at = ?,
                lease_owner = NULL, lease_until = NULL, updated_at = ?
            WHERE id = ?
            """,
            (status, attempts, error[:500], now + backoff_s * attempts, now, job_id),
        )

    def retry_dead(self, job_id: int) -> bool:
        now = time.time()
        updated = self.conn.execute(
            """
            UPDATE sync_jobs
            SET status = 'queued', attempts = 0, next_attempt_at = ?, last_error = NULL, updated_at = ?
            WHERE id = ? AND status = 'dead'
            """,
            (now, now, job_id),
        )
        return updated.rowcount == 1

    def job_counts(self) -> Dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS n FROM sync_jobs GROUP BY status").fetchall()
        counts = {row["status"]: int(row["n"]) for row in rows}
        for name in ("queued", "leased", "done", "dead", "superseded"):
            counts.setdefault(name, 0)
        return counts

    def get_page_state(self, page_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM page_state WHERE page_id = ?", (page_id,)).fetchone()

    def active_revision_map(self) -> Dict[int, str]:
        rows = self.conn.execute(
            "SELECT page_id, active_revision FROM page_state WHERE status = 'published' AND active_revision IS NOT NULL"
        ).fetchall()
        return {int(row["page_id"]): str(row["active_revision"]) for row in rows}

    def list_page_ids(self) -> List[int]:
        rows = self.conn.execute("SELECT page_id FROM page_state WHERE status != 'tombstone'").fetchall()
        return [int(row["page_id"]) for row in rows]

    def stage_revision(
        self,
        *,
        revision_id: str,
        page: Any,
        parents: Sequence[Any],
        children: Sequence[Any],
        content_hash: str,
        metadata_hash: str,
        generation: int,
    ) -> None:
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO revisions(revision_id, page_id, state, generation, content_hash, metadata_hash, page_json, updated_at)
                VALUES (?, ?, 'prepared', ?, ?, ?, ?, ?)
                """,
                (revision_id, page.page_id, generation, content_hash, metadata_hash, page.model_dump_json(), time.time()),
            )
            self.conn.executemany(
                "INSERT INTO parent_records(parent_id, page_id, revision_id, heading, body, ordinal) VALUES (?, ?, ?, ?, ?, ?)",
                [(item.parent_id, page.page_id, revision_id, item.heading, item.text, item.ordinal) for item in parents],
            )
            self.conn.executemany(
                """
                INSERT INTO chunk_records(chunk_id, page_id, revision_id, parent_id, heading, body, embed_text, ordinal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (item.chunk_id, page.page_id, revision_id, item.parent_id, item.heading, item.text, item.embed_text, item.ordinal)
                    for item in children
                ],
            )

    def write_fts(self, revision_id: str, page: Any = None) -> None:
        if page is None:
            stored = self.conn.execute("SELECT page_json FROM revisions WHERE revision_id = ?", (revision_id,)).fetchone()
            if stored is not None and stored["page_json"]:
                page = json.loads(stored["page_json"])
        meta = page if isinstance(page, dict) else (page.model_dump() if page is not None else {})
        title = str(meta.get("name") or "")
        location = location_text(meta.get("book_name") or "", meta.get("chapter_name") or "", meta.get("shelf_names") or [])
        rows = self.conn.execute(
            "SELECT chunk_id, page_id, revision_id, heading, body FROM chunk_records WHERE revision_id = ?",
            (revision_id,),
        ).fetchall()
        with self.transaction():
            for row in rows:
                inserted = self.conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, page_id, revision_id, body, title, location) VALUES (?, ?, ?, ?, ?, ?)",
                    (row["chunk_id"], str(row["page_id"]), row["revision_id"], f"{row['heading']}\n{row['body']}", title, location),
                )
                self.conn.execute(
                    "UPDATE chunk_records SET fts_rowid = ? WHERE chunk_id = ?",
                    (inserted.lastrowid, row["chunk_id"]),
                )

    def _delete_fts(self, where: str, params: Sequence[Any]) -> None:
        self.conn.execute(
            f"DELETE FROM chunks_fts WHERE rowid IN (SELECT fts_rowid FROM chunk_records WHERE fts_rowid IS NOT NULL AND {where})",
            list(params),
        )

    def update_fts_labels(self, page_id: int, title: str, location: str) -> None:
        """Rewrite the title and location columns of a page's active chunks (FTS5 UPDATE by rowid)."""
        self.conn.execute(
            """
            UPDATE chunks_fts SET title = ?, location = ?
            WHERE rowid IN (
                SELECT c.fts_rowid FROM chunk_records c
                JOIN page_state p ON p.page_id = c.page_id AND p.active_revision = c.revision_id
                WHERE c.page_id = ? AND c.fts_rowid IS NOT NULL
            )
            """,
            (title, location, page_id),
        )

    def mark_revision(self, revision_id: str, state: str) -> None:
        self.conn.execute(
            "UPDATE revisions SET state = ?, updated_at = ? WHERE revision_id = ?",
            (state, time.time(), revision_id),
        )

    def discard_revision(self, revision_id: str) -> List[str]:
        rows = self.conn.execute("SELECT chunk_id FROM chunk_records WHERE revision_id = ?", (revision_id,)).fetchall()
        ids = [row["chunk_id"] for row in rows]
        with self.transaction():
            self._delete_fts("revision_id = ?", (revision_id,))
            self.conn.execute("DELETE FROM chunk_records WHERE revision_id = ?", (revision_id,))
            self.conn.execute("DELETE FROM parent_records WHERE revision_id = ?", (revision_id,))
            self.conn.execute(
                "UPDATE revisions SET state = 'failed', updated_at = ? WHERE revision_id = ?",
                (time.time(), revision_id),
            )
        return ids

    def publish(self, page: Any, revision_id: str, content_hash: str, metadata_hash: str, generation: int, schema_version: str, model_id: str) -> Optional[List[str]]:
        now = time.time()
        with self.transaction():
            state = self.conn.execute("SELECT generation, status FROM page_state WHERE page_id = ?", (page.page_id,)).fetchone()
            if state and int(state["generation"]) > generation:
                return None
            if state and state["status"] == "tombstone" and int(state["generation"]) >= generation:
                return None
            old_rows = self.conn.execute(
                "SELECT chunk_id, revision_id FROM chunk_records WHERE page_id = ? AND revision_id != ?",
                (page.page_id, revision_id),
            ).fetchall()
            old_ids = [row["chunk_id"] for row in old_rows]
            if old_ids:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO vector_gc(chunk_id, revision_id) VALUES (?, ?)",
                    [(row["chunk_id"], row["revision_id"]) for row in old_rows],
                )
            shelf_json = json.dumps(page.shelf_names)
            self.conn.execute(
                """
                INSERT INTO page_state(
                    page_id, source_updated_at, content_hash, metadata_hash, active_revision,
                    chunk_schema_version, embedding_model_id, status, generation, book_id, book_name,
                    chapter_name, shelf_names, title, url, tags_str, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'published', ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(page_id) DO UPDATE SET
                    source_updated_at=excluded.source_updated_at,
                    content_hash=excluded.content_hash,
                    metadata_hash=excluded.metadata_hash,
                    active_revision=excluded.active_revision,
                    chunk_schema_version=excluded.chunk_schema_version,
                    embedding_model_id=excluded.embedding_model_id,
                    status='published',
                    generation=excluded.generation,
                    book_id=excluded.book_id,
                    book_name=excluded.book_name,
                    chapter_name=excluded.chapter_name,
                    shelf_names=excluded.shelf_names,
                    title=excluded.title,
                    url=excluded.url,
                    tags_str=excluded.tags_str,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    page.page_id,
                    page.updated_at,
                    content_hash,
                    metadata_hash,
                    revision_id,
                    schema_version,
                    model_id,
                    generation,
                    page.book_id,
                    page.book_name,
                    page.chapter_name,
                    shelf_json,
                    page.name,
                    page.url,
                    page.tags_str,
                    now,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO catalog_pages(page_id, title, url, book_id, book_name, chapter_id, chapter_name, shelf_names, tags_str, revision_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(page_id) DO UPDATE SET
                    title=excluded.title, url=excluded.url, book_id=excluded.book_id, book_name=excluded.book_name,
                    chapter_id=excluded.chapter_id, chapter_name=excluded.chapter_name, shelf_names=excluded.shelf_names,
                    tags_str=excluded.tags_str, revision_id=excluded.revision_id
                """,
                (
                    page.page_id,
                    page.name,
                    page.url,
                    page.book_id,
                    page.book_name,
                    page.chapter_id,
                    page.chapter_name,
                    shelf_json,
                    page.tags_str,
                    revision_id,
                ),
            )
            self.conn.execute("UPDATE revisions SET state = 'published', updated_at = ? WHERE revision_id = ?", (now, revision_id))
            self.conn.execute(
                "UPDATE revisions SET state = 'retired', updated_at = ? WHERE page_id = ? AND revision_id != ? AND state = 'published'",
                (now, page.page_id, revision_id),
            )
            self._delete_fts("page_id = ? AND revision_id != ?", (page.page_id, revision_id))
            self.conn.execute("DELETE FROM chunk_records WHERE page_id = ? AND revision_id != ?", (page.page_id, revision_id))
            self.conn.execute("DELETE FROM parent_records WHERE page_id = ? AND revision_id != ?", (page.page_id, revision_id))
        return old_ids

    def update_metadata_only(self, page: Any, metadata_hash: str, generation: int) -> bool:
        state = self.get_page_state(page.page_id)
        if state is None or state["status"] != "published":
            return False
        if int(state["generation"]) > generation:
            return False
        shelf_json = json.dumps(page.shelf_names)
        now = time.time()
        with self.transaction():
            self.conn.execute(
                """
                UPDATE page_state
                SET metadata_hash = ?, generation = ?, book_id = ?, book_name = ?, chapter_name = ?,
                    shelf_names = ?, title = ?, url = ?, tags_str = ?, source_updated_at = ?, updated_at = ?
                WHERE page_id = ?
                """,
                (
                    metadata_hash,
                    generation,
                    page.book_id,
                    page.book_name,
                    page.chapter_name,
                    shelf_json,
                    page.name,
                    page.url,
                    page.tags_str,
                    page.updated_at,
                    now,
                    page.page_id,
                ),
            )
            self.conn.execute(
                """
                UPDATE catalog_pages
                SET title = ?, url = ?, book_id = ?, book_name = ?, chapter_id = ?, chapter_name = ?, shelf_names = ?, tags_str = ?
                WHERE page_id = ?
                """,
                (
                    page.name,
                    page.url,
                    page.book_id,
                    page.book_name,
                    page.chapter_id,
                    page.chapter_name,
                    shelf_json,
                    page.tags_str,
                    page.page_id,
                ),
            )
            self.update_fts_labels(page.page_id, page.name, location_text(page.book_name, page.chapter_name, page.shelf_names))
        return True

    def tombstone(self, page_id: int, generation: int) -> List[str]:
        rows = self.conn.execute("SELECT chunk_id, revision_id FROM chunk_records WHERE page_id = ?", (page_id,)).fetchall()
        ids = [row["chunk_id"] for row in rows]
        now = time.time()
        with self.transaction():
            state = self.conn.execute("SELECT generation FROM page_state WHERE page_id = ?", (page_id,)).fetchone()
            if state and int(state["generation"]) > generation:
                return []
            if ids:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO vector_gc(chunk_id, revision_id) VALUES (?, ?)",
                    [(row["chunk_id"], row["revision_id"]) for row in rows],
                )
            self._delete_fts("page_id = ?", (page_id,))
            self.conn.execute("DELETE FROM chunk_records WHERE page_id = ?", (page_id,))
            self.conn.execute("DELETE FROM parent_records WHERE page_id = ?", (page_id,))
            self.conn.execute("DELETE FROM catalog_pages WHERE page_id = ?", (page_id,))
            self.conn.execute(
                """
                INSERT INTO page_state(page_id, status, generation, active_revision, updated_at)
                VALUES (?, 'tombstone', ?, NULL, ?)
                ON CONFLICT(page_id) DO UPDATE SET
                    status = 'tombstone', generation = excluded.generation, active_revision = NULL, updated_at = excluded.updated_at
                """,
                (page_id, generation, now),
            )
        return ids

    def gc_ids(self) -> List[str]:
        return [row["chunk_id"] for row in self.conn.execute("SELECT chunk_id FROM vector_gc").fetchall()]

    def clear_gc(self, chunk_ids: Iterable[str]) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        self.conn.executemany("DELETE FROM vector_gc WHERE chunk_id = ?", [(item,) for item in ids])

    def incomplete_revisions(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM revisions WHERE state IN ('prepared', 'indexed')").fetchall()

    def lexical_search(self, match: str, limit: int, page_ids: Optional[Sequence[int]] = None) -> List[sqlite3.Row]:
        """One FTS query. An ACL list of any size is passed as a single JSON parameter."""
        if not match or limit <= 0:
            return []
        if page_ids is not None and len(page_ids) == 0:
            return []
        weights = ", ".join(str(weight) for weight in FTS_WEIGHTS)
        if page_ids is None:
            # Rank inside FTS first and join only the head: joining every
            # match of a common word to page_state costs more than the ranking.
            # Superseded revisions are deleted on publish, so the head rarely
            # loses rows; if it does, fall back to the full join below.
            head = limit * 4
            rows = self._read(
                f"""
                SELECT f.chunk_id AS chunk_id, f.page_id AS page_id, f.revision_id AS revision_id, f.score AS score
                FROM (
                    SELECT chunk_id, page_id, revision_id, bm25(chunks_fts, {weights}) AS score
                    FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?
                ) f
                JOIN page_state p
                  ON p.page_id = CAST(f.page_id AS INTEGER)
                 AND p.active_revision = f.revision_id
                 AND p.status = 'published'
                ORDER BY f.score
                LIMIT ?
                """,
                [match, head, limit],
            )
            if len(rows) >= limit or not self._read(
                "SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1 OFFSET ?", [match, head]
            ):
                return rows
        acl_sql, acl_params = _acl_clause(page_ids, "p.page_id")
        return self._read(
            f"""
            SELECT chunks_fts.chunk_id AS chunk_id, chunks_fts.page_id AS page_id,
                   chunks_fts.revision_id AS revision_id, bm25(chunks_fts, {weights}) AS score
            FROM chunks_fts
            JOIN page_state p
              ON p.page_id = CAST(chunks_fts.page_id AS INTEGER)
             AND p.active_revision = chunks_fts.revision_id
             AND p.status = 'published'
            WHERE chunks_fts MATCH ?
            {acl_sql}
            ORDER BY score
            LIMIT ?
            """,
            [match, *acl_params, limit],
        )

    def search_rows(self, chunk_ids: Sequence[str]) -> Dict[str, sqlite3.Row]:
        """Chunks of active revisions of published pages, with page labels, in one query."""
        if not chunk_ids:
            return {}
        found: Dict[str, sqlite3.Row] = {}
        for batch in _batches(list(chunk_ids)):
            placeholders = ",".join("?" for _ in batch)
            for row in self._read(
                f"""
                SELECT c.chunk_id, c.page_id, c.revision_id, c.parent_id, c.heading, c.body,
                       p.title, p.url, p.book_name, p.chapter_name, p.shelf_names
                FROM chunk_records c
                CROSS JOIN page_state p
                WHERE c.chunk_id IN ({placeholders})
                  AND p.page_id = c.page_id
                  AND p.active_revision = c.revision_id
                  AND p.status = 'published'
                """,
                batch,
            ):
                found[row["chunk_id"]] = row
        return found

    def published_count(self) -> int:
        row = self._read("SELECT COUNT(*) AS n FROM page_state WHERE status = 'published'", [])
        return int(row[0]["n"]) if row else 0

    def _read(self, sql: str, params: Sequence[Any]) -> List[sqlite3.Row]:
        return self._reader().execute(sql, list(params)).fetchall()

    def book_pages_to_reembed(self, book_id: int, book_name: str) -> List[int]:
        """Pages of a book whose indexed book name differs; the name is part of their embeddings."""
        rows = self.conn.execute(
            "SELECT page_id FROM page_state WHERE book_id = ? AND status = 'published' AND COALESCE(book_name, '') != ?",
            (book_id, book_name),
        ).fetchall()
        return [int(row["page_id"]) for row in rows]

    def relabel_book(self, book_id: int, book_name: str, shelf_names: Sequence[str]) -> int:
        """Update book name and shelves in state, catalog, and FTS labels. Embeddings are not rewritten here."""
        shelf_json = json.dumps(list(shelf_names), ensure_ascii=False)
        with self.transaction():
            updated = self.conn.execute(
                "UPDATE page_state SET book_name = ?, shelf_names = ? WHERE book_id = ?",
                (book_name, shelf_json, book_id),
            )
            self.conn.execute(
                "UPDATE catalog_pages SET book_name = ?, shelf_names = ? WHERE book_id = ?",
                (book_name, shelf_json, book_id),
            )
            pages = self.conn.execute(
                "SELECT page_id, title, chapter_name FROM page_state WHERE book_id = ? AND status = 'published'",
                (book_id,),
            ).fetchall()
            for page in pages:
                self.update_fts_labels(
                    int(page["page_id"]),
                    str(page["title"] or ""),
                    location_text(book_name, page["chapter_name"] or "", shelf_names),
                )
        return int(updated.rowcount)

    def catalog_page_ids_for(self, book_id: Optional[int] = None, chapter_id: Optional[int] = None, shelf_name: str = "") -> List[int]:
        """Indexed pages in a book, chapter, or shelf, regardless of ACL (for index maintenance only)."""
        clauses, params = [], []
        if book_id is not None:
            clauses.append("c.book_id = ?")
            params.append(int(book_id))
        if chapter_id is not None:
            clauses.append("c.chapter_id = ?")
            params.append(int(chapter_id))
        if shelf_name:
            clauses.append("EXISTS (SELECT 1 FROM json_each(c.shelf_names) s WHERE s.value = ?)")
            params.append(shelf_name)
        if not clauses:
            return []
        rows = self.conn.execute(f"SELECT c.page_id FROM catalog_pages c WHERE {' AND '.join(clauses)}", params).fetchall()
        return [int(row["page_id"]) for row in rows]

    def catalog_book_ids_for_shelf(self, shelf_name: str) -> List[int]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT c.book_id FROM catalog_pages c
            WHERE EXISTS (SELECT 1 FROM json_each(c.shelf_names) s WHERE s.value = ?)
            """,
            (shelf_name,),
        ).fetchall()
        return [int(row["book_id"] or 0) for row in rows if row["book_id"]]

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        found: Dict[str, sqlite3.Row] = {}
        for batch in _batches(list(chunk_ids)):
            placeholders = ",".join("?" for _ in batch)
            for row in self.conn.execute(f"SELECT * FROM chunk_records WHERE chunk_id IN ({placeholders})", batch).fetchall():
                found[row["chunk_id"]] = row
        return found

    def get_parent(self, parent_id: str, revision_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM parent_records WHERE parent_id = ? AND revision_id = ?",
            (parent_id, revision_id),
        ).fetchone()

    def page_chunks(self, page_id: int, revision_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunk_records WHERE page_id = ? AND revision_id = ? ORDER BY ordinal",
            (page_id, revision_id),
        ).fetchall()

    def page_parents(self, page_id: int, revision_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM parent_records WHERE page_id = ? AND revision_id = ? ORDER BY ordinal",
            (page_id, revision_id),
        ).fetchall()

    # Catalog. Every query takes the caller's ACL (None = all pages) and an
    # optional CatalogFilter; both become single JSON parameters.

    def _catalog_where(self, allowed: Optional[Sequence[int]], flt: Optional[CatalogFilter]) -> tuple:
        acl_sql, params = _acl_clause(allowed, "c.page_id")
        clauses = [acl_sql] if acl_sql else []
        if flt is not None:
            if flt.shelves is not None:
                clauses.append(
                    "AND EXISTS (SELECT 1 FROM json_each(c.shelf_names) s WHERE s.value IN (SELECT value FROM json_each(?)))"
                )
                params.append(json.dumps(list(flt.shelves), ensure_ascii=False))
            if flt.book_ids is not None:
                clauses.append("AND c.book_id IN (SELECT value FROM json_each(?))")
                params.append(json.dumps([int(item) for item in flt.book_ids]))
            if flt.chapter_ids is not None:
                clauses.append("AND c.chapter_id IN (SELECT value FROM json_each(?))")
                params.append(json.dumps([int(item) for item in flt.chapter_ids]))
        return " ".join(clauses), params

    def catalog_rows(self, allowed: Optional[Sequence[int]], offset: int, limit: int) -> List[sqlite3.Row]:
        if allowed is not None and not allowed:
            return []
        where, params = self._catalog_where(allowed, None)
        return self._read(
            f"SELECT * FROM catalog_pages c WHERE 1 = 1 {where} ORDER BY c.book_name, c.title LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )

    def catalog_counts(self, allowed: Optional[Sequence[int]], flt: Optional[CatalogFilter] = None) -> Dict[str, Any]:
        if allowed is not None and len(allowed) == 0:
            return {"pages": 0, "books": 0, "chapters": 0, "book_pages": {}, "shelves": []}
        where, params = self._catalog_where(allowed, flt)
        totals = self._read(
            f"""
            SELECT COUNT(*) AS pages,
                   COUNT(DISTINCT CASE WHEN c.chapter_id != 0 THEN c.chapter_id END) AS chapters
            FROM catalog_pages c WHERE 1 = 1 {where}
            """,
            params,
        )[0]
        book_pages: Dict[str, Dict[str, Any]] = {}
        for row in self._read(
            f"""
            SELECT c.book_id, COALESCE(c.book_name, 'General Library') AS book_name, COUNT(*) AS n
            FROM catalog_pages c WHERE 1 = 1 {where}
            GROUP BY c.book_id, c.book_name
            """,
            params,
        ):
            _remember_book(book_pages, row["book_id"], row["book_name"], row["n"])
        shelves = [
            row["shelf"]
            for row in self._read(
                f"""
                SELECT DISTINCT s.value AS shelf
                FROM catalog_pages c, json_each(c.shelf_names) s
                WHERE 1 = 1 {where}
                ORDER BY shelf
                """,
                params,
            )
            if row["shelf"]
        ]
        return {
            "pages": int(totals["pages"] or 0),
            "books": len(book_pages),
            "chapters": int(totals["chapters"] or 0),
            "book_pages": book_pages,
            "shelves": shelves,
        }

    def catalog_books(
        self, allowed: Optional[Sequence[int]], offset: int, limit: int, flt: Optional[CatalogFilter] = None
    ) -> List[Dict[str, Any]]:
        return self.catalog_book_page(allowed, offset, limit, flt)["items"]

    def catalog_book_page(
        self, allowed: Optional[Sequence[int]], offset: int, limit: int, flt: Optional[CatalogFilter] = None
    ) -> Dict[str, Any]:
        if limit <= 0 or (allowed is not None and len(allowed) == 0):
            return {"total": 0, "items": []}
        where, params = self._catalog_where(allowed, flt)
        total = self._read(f"SELECT COUNT(DISTINCT c.book_id) AS n FROM catalog_pages c WHERE 1 = 1 {where}", params)[0]["n"]
        rows = self._read(
            f"""
            SELECT c.book_id,
                   COALESCE(MAX(c.book_name), 'General Library') AS book_name,
                   COUNT(*) AS page_count,
                   COUNT(DISTINCT CASE WHEN c.chapter_id != 0 THEN c.chapter_id END) AS chapter_count,
                   MAX(c.shelf_names) AS shelf_names
            FROM catalog_pages c WHERE 1 = 1 {where}
            GROUP BY c.book_id
            ORDER BY book_name, c.book_id
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        )
        return {
            "total": int(total or 0),
            "items": [
                {
                    "book_id": int(row["book_id"] or 0),
                    "book_name": row["book_name"],
                    "page_count": int(row["page_count"]),
                    "chapter_count": int(row["chapter_count"]),
                    "shelf_names": _shelf_names(row["shelf_names"]),
                }
                for row in rows
            ],
        }

    def catalog_shelf_page(
        self, allowed: Optional[Sequence[int]], offset: int, limit: int, flt: Optional[CatalogFilter] = None
    ) -> Dict[str, Any]:
        if limit <= 0 or (allowed is not None and len(allowed) == 0):
            return {"total": 0, "items": [], "books_without_shelf": 0}
        where, params = self._catalog_where(allowed, flt)
        total = self._read(
            f"SELECT COUNT(DISTINCT s.value) AS n FROM catalog_pages c, json_each(c.shelf_names) s WHERE 1 = 1 {where}",
            params,
        )[0]["n"]
        rows = self._read(
            f"""
            SELECT s.value AS shelf_name, COUNT(DISTINCT c.book_id) AS book_count, COUNT(*) AS page_count
            FROM catalog_pages c, json_each(c.shelf_names) s
            WHERE 1 = 1 {where}
            GROUP BY s.value
            ORDER BY s.value
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        )
        loose = self._read(
            f"""
            SELECT COUNT(DISTINCT c.book_id) AS n FROM catalog_pages c
            WHERE json_array_length(COALESCE(c.shelf_names, '[]')) = 0 {where}
            """,
            params,
        )[0]["n"]
        return {
            "total": int(total or 0),
            "items": [
                {"shelf_name": row["shelf_name"], "book_count": int(row["book_count"]), "page_count": int(row["page_count"])}
                for row in rows
            ],
            "books_without_shelf": int(loose or 0),
        }

    def catalog_chapter_page(
        self, allowed: Optional[Sequence[int]], offset: int, limit: int, flt: Optional[CatalogFilter] = None
    ) -> Dict[str, Any]:
        if limit <= 0 or (allowed is not None and len(allowed) == 0):
            return {"total": 0, "items": [], "pages_outside_chapters": 0}
        where, params = self._catalog_where(allowed, flt)
        total = self._read(
            f"SELECT COUNT(DISTINCT c.chapter_id) AS n FROM catalog_pages c WHERE c.chapter_id != 0 {where}",
            params,
        )[0]["n"]
        rows = self._read(
            f"""
            SELECT c.chapter_id, MAX(c.chapter_name) AS chapter_name, MAX(c.book_id) AS book_id,
                   COALESCE(MAX(c.book_name), 'General Library') AS book_name, COUNT(*) AS page_count
            FROM catalog_pages c
            WHERE c.chapter_id != 0 {where}
            GROUP BY c.chapter_id
            ORDER BY book_name, chapter_name, c.chapter_id
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        )
        loose = self._read(
            f"SELECT COUNT(*) AS n FROM catalog_pages c WHERE COALESCE(c.chapter_id, 0) = 0 {where}",
            params,
        )[0]["n"]
        return {
            "total": int(total or 0),
            "items": [
                {
                    "chapter_id": int(row["chapter_id"]),
                    "chapter_name": row["chapter_name"],
                    "book_id": int(row["book_id"] or 0),
                    "book_name": row["book_name"],
                    "page_count": int(row["page_count"]),
                }
                for row in rows
            ],
            "pages_outside_chapters": int(loose or 0),
        }

    def catalog_page_page(
        self, allowed: Optional[Sequence[int]], offset: int, limit: int, flt: Optional[CatalogFilter] = None
    ) -> Dict[str, Any]:
        if limit <= 0 or (allowed is not None and len(allowed) == 0):
            return {"total": 0, "items": []}
        where, params = self._catalog_where(allowed, flt)
        total = self._read(f"SELECT COUNT(*) AS n FROM catalog_pages c WHERE 1 = 1 {where}", params)[0]["n"]
        rows = self._read(
            f"""
            SELECT c.page_id, c.title, c.url, c.book_name, c.chapter_id, c.chapter_name
            FROM catalog_pages c WHERE 1 = 1 {where}
            ORDER BY c.book_name, (COALESCE(c.chapter_id, 0) = 0), c.chapter_name, c.title, c.page_id
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        )
        return {
            "total": int(total or 0),
            "items": [
                {
                    "page_id": int(row["page_id"]),
                    "title": row["title"] or "",
                    "url": row["url"] or "",
                    "book_name": real_name(row["book_name"]),
                    "chapter_name": real_name(row["chapter_name"]) if row["chapter_id"] else "",
                }
                for row in rows
            ],
        }

    def catalog_page_ids(self, allowed: Optional[Sequence[int]], flt: CatalogFilter) -> List[int]:
        if allowed is not None and len(allowed) == 0:
            return []
        where, params = self._catalog_where(allowed, flt)
        return [int(row["page_id"]) for row in self._read(f"SELECT c.page_id FROM catalog_pages c WHERE 1 = 1 {where}", params)]

    def catalog_shelf_names(self, allowed: Optional[Sequence[int]]) -> List[str]:
        where, params = self._catalog_where(allowed, None)
        rows = self._read(
            f"SELECT DISTINCT s.value AS shelf FROM catalog_pages c, json_each(c.shelf_names) s WHERE 1 = 1 {where}",
            params,
        )
        return [str(row["shelf"]) for row in rows if row["shelf"]]

    def catalog_book_names(self, allowed: Optional[Sequence[int]]) -> List[tuple]:
        where, params = self._catalog_where(allowed, None)
        rows = self._read(
            f"SELECT DISTINCT c.book_id, c.book_name FROM catalog_pages c WHERE c.book_id != 0 {where}",
            params,
        )
        return [(int(row["book_id"]), str(row["book_name"] or "")) for row in rows if real_name(row["book_name"])]

    def catalog_chapter_names(self, allowed: Optional[Sequence[int]], book_ids: Optional[Sequence[int]] = None) -> List[tuple]:
        flt = CatalogFilter(book_ids=list(book_ids)) if book_ids is not None else None
        where, params = self._catalog_where(allowed, flt)
        rows = self._read(
            f"""
            SELECT DISTINCT c.chapter_id, c.chapter_name, c.book_name
            FROM catalog_pages c WHERE c.chapter_id != 0 {where}
            """,
            params,
        )
        return [
            (int(row["chapter_id"]), str(row["chapter_name"] or ""), str(row["book_name"] or ""))
            for row in rows
            if real_name(row["chapter_name"])
        ]

    def log_shadow(self, query_fp: str, legacy_pages: List[int], adaptive_pages: List[int], latency_ms: float) -> None:
        self.conn.execute(
            "INSERT INTO shadow_log(created_at, query_fp, legacy_pages, adaptive_pages, latency_ms) VALUES (?, ?, ?, ?, ?)",
            (time.time(), query_fp, json.dumps(legacy_pages), json.dumps(adaptive_pages), latency_ms),
        )

    def log_usage(self, usage: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO usage_log(
                created_at, purpose, provider, model, prompt_tokens_est, completion_tokens_est,
                prompt_tokens_actual, completion_tokens_actual, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                usage.purpose,
                usage.provider,
                usage.model,
                usage.prompt_tokens_est,
                usage.completion_tokens_est,
                usage.prompt_tokens_actual,
                usage.completion_tokens_actual,
                usage.latency_ms,
            ),
        )

    def log_assistant_turn(
        self,
        *,
        user_id: str,
        query: str,
        answer: str,
        provider: str,
        model: str,
        reasoning_effort: str,
        input_tokens: int,
        output_tokens: int,
        input_usd_per_mtok: Optional[float],
        output_usd_per_mtok: Optional[float],
        cost_usd: Optional[float],
        evidence_json: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO assistant_turns(
                created_at, user_id, query, answer, provider, model, reasoning_effort,
                input_tokens, output_tokens, input_usd_per_mtok, output_usd_per_mtok, cost_usd,
                evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                user_id,
                query,
                answer,
                provider,
                model,
                reasoning_effort,
                input_tokens,
                output_tokens,
                input_usd_per_mtok,
                output_usd_per_mtok,
                cost_usd,
                evidence_json,
            ),
        )

    def _job(self, row: sqlite3.Row) -> SyncJob:
        payload = json.loads(row["payload_json"] or "{}")
        return SyncJob(
            id=int(row["id"]),
            page_id=int(row["page_id"]),
            event=row["event"],
            generation=int(row["generation"]),
            status=row["status"],
            attempts=int(row["attempts"]),
            payload=payload,
            last_error=row["last_error"] or "",
        )


class _Transaction:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self.conn = conn
        self.lock = lock

    def __enter__(self):
        self.lock.acquire()
        try:
            self.conn.execute("BEGIN IMMEDIATE")
        except Exception:
            self.lock.release()
            raise
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute("ROLLBACK")
        finally:
            self.lock.release()
        return False


def _remember_book(books: Dict[str, Dict[str, Any]], book_id: Any, book_name: str, count: Any) -> None:
    name = book_name or "General Library"
    key = f"{int(book_id or 0)}:{name}"
    current = books.get(key)
    if current is None:
        books[key] = {"book_id": int(book_id or 0), "book_name": name, "pages": int(count)}
        return
    current["pages"] += int(count)


def _shelf_names(raw: Optional[str]) -> List[str]:
    try:
        names = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [str(name) for name in names if name]


def _acl_clause(page_ids: Optional[Sequence[int]], column: str) -> tuple:
    """`AND column IN (json list)`; None means unrestricted."""
    if page_ids is None:
        return "", []
    return f"AND {column} IN (SELECT value FROM json_each(?))", [json.dumps([int(item) for item in page_ids])]


def _batches(items: List[Any], size: int = 500) -> Iterable[List[Any]]:
    """Primary-key IN lists stay small enough for SQLite's variable limit and use the index."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
