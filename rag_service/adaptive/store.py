import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from adaptive.contracts import AuthorizationScope, SyncJob


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
                cost_usd REAL
            );
            """
        )
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                page_id UNINDEXED,
                revision_id UNINDEXED,
                body,
                tokenize = "unicode61 remove_diacritics 2"
            )
            """
        )

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

    def write_fts(self, revision_id: str) -> None:
        rows = self.conn.execute(
            "SELECT chunk_id, page_id, revision_id, heading, body FROM chunk_records WHERE revision_id = ?",
            (revision_id,),
        ).fetchall()
        with self.transaction():
            self.conn.executemany(
                "INSERT INTO chunks_fts(chunk_id, page_id, revision_id, body) VALUES (?, ?, ?, ?)",
                [
                    (row["chunk_id"], str(row["page_id"]), row["revision_id"], f"{row['heading']}\n{row['body']}")
                    for row in rows
                ],
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
            self.conn.execute("DELETE FROM chunks_fts WHERE revision_id = ?", (revision_id,))
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
            self.conn.execute("DELETE FROM chunks_fts WHERE page_id = ? AND revision_id != ?", (str(page.page_id), revision_id))
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
            self.conn.execute("DELETE FROM chunks_fts WHERE page_id = ?", (str(page_id),))
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
        if not match or limit <= 0:
            return []
        if page_ids is not None and len(page_ids) == 0:
            return []
        if page_ids is None:
            return self._lexical_query(match, limit, None)
        found: List[sqlite3.Row] = []
        ids = list(page_ids)
        for start in range(0, len(ids), 200):
            found.extend(self._lexical_query(match, limit, ids[start : start + 200]))
        found.sort(key=lambda row: row["score"])
        chosen = []
        seen = set()
        for row in found:
            if row["chunk_id"] in seen:
                continue
            seen.add(row["chunk_id"])
            chosen.append(row)
            if len(chosen) >= limit:
                break
        return chosen

    def _lexical_query(self, match: str, limit: int, page_ids: Optional[Sequence[int]]) -> List[sqlite3.Row]:
        acl_sql = ""
        params: List[Any] = [match]
        if page_ids is not None:
            placeholders = ",".join("?" for _ in page_ids)
            acl_sql = f" AND p.page_id IN ({placeholders})"
            params.extend(int(page_id) for page_id in page_ids)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT chunks_fts.chunk_id AS chunk_id, chunks_fts.page_id AS page_id,
                   chunks_fts.revision_id AS revision_id, bm25(chunks_fts) AS score
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
            params,
        ).fetchall()

    def relabel_book(self, book_id: int, book_name: str, shelf_names: Sequence[str]) -> int:
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
        return int(updated.rowcount)

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"SELECT * FROM chunk_records WHERE chunk_id IN ({placeholders})",
            list(chunk_ids),
        ).fetchall()
        return {row["chunk_id"]: row for row in rows}

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

    def catalog_rows(self, allowed: Optional[Sequence[int]], offset: int, limit: int) -> List[sqlite3.Row]:
        if allowed is None:
            return self.conn.execute(
                "SELECT * FROM catalog_pages ORDER BY book_name, title LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        if not allowed:
            return []
        rows = []
        ids = list(allowed)
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self.conn.execute(
                    f"SELECT * FROM catalog_pages WHERE page_id IN ({placeholders}) ORDER BY book_name, title",
                    batch,
                ).fetchall()
            )
        rows.sort(key=lambda row: (row["book_name"] or "", row["title"] or ""))
        return rows[offset : offset + limit]

    def catalog_counts(self, allowed: Optional[Sequence[int]]) -> Dict[str, Any]:
        if allowed is not None and len(allowed) == 0:
            return {"pages": 0, "books": 0, "book_pages": {}, "shelves": []}
        book_pages: Dict[str, int] = {}
        shelves = set()
        page_total = 0
        if allowed is None:
            page_total = int(self.conn.execute("SELECT COUNT(*) AS n FROM catalog_pages").fetchone()["n"])
            for row in self.conn.execute(
                """
                SELECT book_id, COALESCE(book_name, 'General Library') AS book_name, COUNT(*) AS n
                FROM catalog_pages
                GROUP BY book_id, book_name
                """
            ).fetchall():
                _remember_book(book_pages, row["book_id"], row["book_name"], row["n"])
            for row in self.conn.execute("SELECT shelf_names FROM catalog_pages").fetchall():
                shelves.update(_shelf_names(row["shelf_names"]))
            return {"pages": page_total, "books": len(book_pages), "book_pages": book_pages, "shelves": sorted(shelves)}

        ids = list(allowed)
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            placeholders = ",".join("?" for _ in batch)
            page_total += int(
                self.conn.execute(
                    f"SELECT COUNT(*) AS n FROM catalog_pages WHERE page_id IN ({placeholders})",
                    batch,
                ).fetchone()["n"]
            )
            for row in self.conn.execute(
                f"""
                SELECT book_id, COALESCE(book_name, 'General Library') AS book_name, COUNT(*) AS n
                FROM catalog_pages
                WHERE page_id IN ({placeholders})
                GROUP BY book_id, book_name
                """,
                batch,
            ).fetchall():
                _remember_book(book_pages, row["book_id"], row["book_name"], row["n"])
            for row in self.conn.execute(
                f"SELECT shelf_names FROM catalog_pages WHERE page_id IN ({placeholders})",
                batch,
            ).fetchall():
                shelves.update(_shelf_names(row["shelf_names"]))
        return {"pages": page_total, "books": len(book_pages), "book_pages": book_pages, "shelves": sorted(shelves)}

    def catalog_books(self, allowed: Optional[Sequence[int]], offset: int, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        if allowed is not None and len(allowed) == 0:
            return []
        if allowed is None:
            rows = self.conn.execute(
                """
                SELECT book_id,
                       COALESCE(book_name, 'General Library') AS book_name,
                       COUNT(*) AS page_count
                FROM catalog_pages
                GROUP BY book_id, book_name
                ORDER BY book_name
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            return [
                {"book_id": int(row["book_id"] or 0), "book_name": row["book_name"], "page_count": int(row["page_count"])}
                for row in rows
            ]
        merged: Dict[int, Dict[str, Any]] = {}
        ids = list(allowed)
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            placeholders = ",".join("?" for _ in batch)
            for row in self.conn.execute(
                f"""
                SELECT book_id,
                       COALESCE(book_name, 'General Library') AS book_name,
                       COUNT(*) AS page_count
                FROM catalog_pages
                WHERE page_id IN ({placeholders})
                GROUP BY book_id, book_name
                """,
                batch,
            ).fetchall():
                book_id = int(row["book_id"] or 0)
                current = merged.setdefault(book_id, {"book_id": book_id, "book_name": row["book_name"], "page_count": 0})
                current["page_count"] += int(row["page_count"])
                if row["book_name"]:
                    current["book_name"] = row["book_name"]
        ordered = sorted(merged.values(), key=lambda item: (item["book_name"] or "", item["book_id"]))
        return ordered[offset : offset + limit]

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
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO assistant_turns(
                created_at, user_id, query, answer, provider, model, reasoning_effort,
                input_tokens, output_tokens, input_usd_per_mtok, output_usd_per_mtok, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
