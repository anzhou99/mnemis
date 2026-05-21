# core/eval/prompt_manager.py
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_DB_PATH = Path("./mnemis_eval.db")   # 复用 Eval 数据库


class PromptVersion:
    """单个 Prompt 版本"""
    def __init__(
        self,
        prompt_id: str,
        name: str,
        content: str,
        version: int,
        eval_score: float | None = None,
        notes: str = "",
        is_active: bool = False,
    ):
        self.prompt_id = prompt_id
        self.name = name
        self.content = content
        self.version = version
        self.eval_score = eval_score
        self.notes = notes
        self.is_active = is_active
        self.created_at = datetime.now()
        self.content_hash = hashlib.md5(content.encode()).hexdigest()[:8]


class PromptManager:
    """
    Prompt 版本管理器。
    核心原则：每次修改都创建新版本，从不覆盖历史，支持回滚到任意版本。
    """

    def __init__(self):
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(PROMPT_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id          TEXT PRIMARY KEY,
                    prompt_id   TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    version     INTEGER NOT NULL,
                    eval_score  REAL,
                    notes       TEXT,
                    is_active   INTEGER DEFAULT 0,
                    content_hash TEXT,
                    created_at  TEXT NOT NULL
                )
            """)
            # 防止完全相同的 prompt 重复写入
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS
                idx_prompt_hash ON prompt_versions(prompt_id, content_hash)
            """)

    def save(
        self,
        prompt_id: str,
        name: str,
        content: str,
        notes: str = "",
        eval_score: float | None = None,
        set_active: bool = False,
    ) -> PromptVersion:
        """
        保存一个新的 Prompt 版本。
        如果内容完全相同（hash 相同），跳过保存。
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

        with sqlite3.connect(PROMPT_DB_PATH) as conn:
            # 检查是否已存在相同内容
            existing = conn.execute(
                "SELECT id FROM prompt_versions WHERE prompt_id=? AND content_hash=?",
                (prompt_id, content_hash)
            ).fetchone()

            if existing:
                logger.debug(f"Prompt {prompt_id} v? already exists (same content)")
                return self.get_active(prompt_id)

            # 获取当前最大版本号
            row = conn.execute(
                "SELECT MAX(version) FROM prompt_versions WHERE prompt_id=?",
                (prompt_id,)
            ).fetchone()
            next_version = (row[0] or 0) + 1

            version_id = f"{prompt_id}-v{next_version}"

            if set_active:
                # 先把其他版本设为非活跃
                conn.execute(
                    "UPDATE prompt_versions SET is_active=0 WHERE prompt_id=?",
                    (prompt_id,)
                )

            conn.execute(
                """INSERT INTO prompt_versions
                   (id, prompt_id, name, content, version, eval_score,
                    notes, is_active, content_hash, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    version_id, prompt_id, name, content, next_version,
                    eval_score, notes, int(set_active), content_hash,
                    datetime.now().isoformat()
                )
            )

        pv = PromptVersion(
            prompt_id=prompt_id, name=name, content=content,
            version=next_version, eval_score=eval_score,
            notes=notes, is_active=set_active
        )
        logger.info(f"Saved prompt {prompt_id} v{next_version} (hash:{content_hash})")
        return pv

    def get_active(self, prompt_id: str) -> PromptVersion | None:
        """获取当前活跃版本"""
        with sqlite3.connect(PROMPT_DB_PATH) as conn:
            row = conn.execute(
                """SELECT * FROM prompt_versions
                   WHERE prompt_id=? AND is_active=1
                   ORDER BY version DESC LIMIT 1""",
                (prompt_id,)
            ).fetchone()
            if not row:
                # 没有活跃版本，取最新版本
                row = conn.execute(
                    """SELECT * FROM prompt_versions
                       WHERE prompt_id=?
                       ORDER BY version DESC LIMIT 1""",
                    (prompt_id,)
                ).fetchone()
        return self._row_to_version(row) if row else None

    def rollback(self, prompt_id: str, version: int) -> bool:
        """回滚到指定版本"""
        with sqlite3.connect(PROMPT_DB_PATH) as conn:
            row = conn.execute(
                "SELECT id FROM prompt_versions WHERE prompt_id=? AND version=?",
                (prompt_id, version)
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE prompt_versions SET is_active=0 WHERE prompt_id=?",
                (prompt_id,)
            )
            conn.execute(
                "UPDATE prompt_versions SET is_active=1 WHERE id=?",
                (row[0],)
            )
        logger.info(f"Rolled back {prompt_id} to v{version}")
        return True

    def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        """列出某个 Prompt 的所有历史版本"""
        with sqlite3.connect(PROMPT_DB_PATH) as conn:
            rows = conn.execute(
                """SELECT * FROM prompt_versions WHERE prompt_id=?
                   ORDER BY version DESC""",
                (prompt_id,)
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def _row_to_version(self, row) -> PromptVersion:
        if isinstance(row, sqlite3.Row):
            d = dict(row)
        else:
            cols = ["id","prompt_id","name","content","version",
                    "eval_score","notes","is_active","content_hash","created_at"]
            d = dict(zip(cols, row))

        pv = PromptVersion(
            prompt_id=d["prompt_id"], name=d["name"],
            content=d["content"], version=d["version"],
            eval_score=d.get("eval_score"), notes=d.get("notes",""),
            is_active=bool(d.get("is_active",0)),
        )
        pv.content_hash = d.get("content_hash","")
        return pv