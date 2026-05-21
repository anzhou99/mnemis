# SQLite 操作封装

import sqlite3
from typing import Literal
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from core.memory.models import Memory, Session, Message, MemoryType
from utils.logger import get_logger

logger = get_logger(__name__)


DB_PATH = Path("./mnemis_memory.db")

SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id        TEXT PRIMARY KEY,
        started_at   TEXT NOT NULL,
        ended_at    TEXT,
        title     TEXT,
        summary    TEXT,
        message_count   INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS messages (
        id            TEXT PRIMARY KEY,
        session_id    TEXT NOT NULL REFERENCES sessions(id),
        role          TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content       TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        token_count   INTEGER
    );

    CREATE TABLE IF NOT EXISTS memories (
        id                TEXT PRIMARY KEY,
        content           TEXT NOT NULL,
        memory_type       TEXT NOT NULL,
        confidence        REAL NOT NULL DEFAULT 1.0,
        source_session_id TEXT,
        created_at        TEXT NOT NULL,
        last_accessed     TEXT,
        access_count      INTEGER DEFAULT 0,
        vector_id         TEXT
    );


    CREATE TABLE IF NOT EXISTS memory_conflicts (
        id              TEXT PRIMARY KEY,
        new_memory_id   TEXT,
        old_memory_id   TEXT,
        conflict_type   TEXT,
        resolution      TEXT,
        resolved_at     TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
"""


class MemoryDatabase:
    """SQLite 操作封装——所有对记忆数据库的读写都通过这个类"""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        logger.debug(f"Memory database initialized: {self.db_path}")

    @contextmanager
    def _conn(self):
        """连接上下文管理器，自动提交或回滚"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 让查询结果支持按列名访问
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Session 操作 ──────────────────────────────────────────────
    def create_session(self) -> Session:
        session = Session(id=str(uuid.uuid4))
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions(id,started_at) VALUES (?, ?)",
                (session.id, session.started_at.isoformat()),
            )
        logger.debug(f"Session created: {session.id}")
        return session

    def end_session(self, session_id: str, title: str, summary: str):
        with self._conn() as conn:
            conn.execute(
                """UPDATE sessions
                    SET ended_at=?, title=?, summary=?,
                        message_count=(SELECT COUNT(*) FROM messages WHERE session_id=?)
                    WHERE id=?
                """,
                (
                    datetime.now().isoformat(),
                    title,
                    summary,
                    session_id,
                    session_id,
                ),
            )

    def get_recent_sessions(self, limit: int = 5) -> list[Session]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE ened_at IS NOT NULL ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    # ── Message 操作 ──────────────────────────────────────────────

    def save_message(
        self,
        sesssion_id: str,
        role: Literal["user", "assistant"],
        content: str,
        token_count: int | None = None,
    ) -> Message:
        msg = Message(
            id=str(uuid.uuid4()),
            session_id=sesssion_id,
            role=role,
            content=content,
            token_count=token_count,
        )

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?)",
                (
                    msg.id,
                    msg.session_id,
                    msg.role,
                    msg.content,
                    msg.created_at.isoformat(),
                    msg.token_count,
                ),
            )
        return msg

    def get_session_messages(self, session_id: str) -> list[Message]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY created_at",
                (session_id),
            ).fetchall()
        return [self._row_to_message(r) for r in rows]

    # ── Memory 操作 ──────────────────────────────────────────────
    def save_memory(self, memory: Memory) -> Memory:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO memories
                (id, content, memory_type, confidence, source_session_id,
                    created_at, last_accessed, access_count, vector_id)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    memory.id,
                    memory.content,
                    memory.memory_type.value,
                    memory.confidence,
                    memory.source_session_id,
                    memory.created_at.isoformat(),
                    (
                        memory.last_accessed.isoformat()
                        if memory.last_accessed
                        else None
                    ),
                    memory.access_count,
                    memory.vector_id,
                ),
            )
        logger.debug(
            f"Memory saved: [{memory.memory_type.value}] {memory.content[:50]}"
        )
        return memory

    def update_memory(self, memory: Memory):
        """更新记忆（置信度、向量ID等）"""
        with self._conn() as conn:
            conn.execute(
                """UPDATE memories
                SET confidence=?, vector_id=?, last_accessed=?, access_count=?
                WHERE id=?""",
                (
                    memory.confidence,
                    memory.vector_id,
                    (
                        memory.last_accessed.isoformat()
                        if memory.last_accessed
                        else None
                    ),
                    memory.access_count,
                    memory.id,
                ),
            )

    def get_memories_by_type(
        self,
        memory_type: MemoryType | None = None,
        min_confidence: float = 0.3,
        limit: int = 20,
    ) -> list[Memory]:
        with self._conn() as conn:
            if memory_type:
                rows = conn.execute(
                    """SELECT * FROM memories
                       WHERE memory_type=? AND confidence>=?
                       ORDER BY confidence DESC, created_at DESC
                       LIMIT ?""",
                    (memory_type.value, min_confidence, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM memories
                       WHERE confidence>=?
                       ORDER BY confidence DESC, created_at DESC
                       LIMIT ?""",
                    (min_confidence, limit),
                ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def delete_memory(self, memory_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            total_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type = conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
            ).fetchall()
        return {
            "total_sessions": total_sessions,
            "total_memories": total_memories,
            "by_type": {r[0]: r[1] for r in by_type},
        }

    # ── 私有辅助方法 ──────────────────────────────────────────────

    @staticmethod
    def _row_to_session(row) -> Session:
        return Session(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(
                datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
            ),
            title=row["title"],
            summary=row["summary"],
            message_count=row["message_count"],
        )

    @staticmethod
    def _row_to_message(row) -> Message:
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
            token_count=row["token_count"],
        )

    @staticmethod
    def _row_to_memory(row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            confidence=row["confidence"],
            source_session_id=row["source_session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
            access_count=row["access_count"],
            vector_id=row["vector_id"],
        )
