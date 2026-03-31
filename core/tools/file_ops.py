import os
from pathlib import Path
from core.tools.base import ToolSchema, ToolResult
from utils.logger import get_logger

logger = get_logger(__name__)

WORKSPACE_DIR = Path("./workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)


# ── Schema 定义 ──────────────────────────────────────────────────

READ_FILE_SCHEMA = ToolSchema(
    name="read_file",
    description="读取工作区内的文件内容。只能读取 workspace/ 目录下的文件。",
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "文件名（不含路径），如 'notes.md' 或 'data.txt'",
            }
        },
        "required": ["filename"],
    },
)

WRITE_FILE_SCHEMA = ToolSchema(
    name="write_file",
    description="将内容写入工作区内的文件。文件不存在时自动创建，已存在时覆盖。只能写入 workspace/ 目录。",
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "文件名（不含路径），如 'notes.md'",
            },
            "content": {"type": "string", "description": "要写入的完整文件内容"},
        },
        "required": ["filename", "content"],
    },
)

# ── 执行函数 ──────────────────────────────────────────────────────


def execute_read_file(filename: str) -> ToolResult:
    # 路径安全检查：防止 ../../../etc/passwd 类型的路径穿越攻击
    safe_path = WORKSPACE_DIR / Path(filename).name
    logger.debug(f"Reading file: {safe_path}")

    try:
        if not safe_path.exists():
            return ToolResult(
                tool_use_id="",
                content=f"文件不存在：{filename}。workspace/ 中现有文件：{list_workspace()}",
                is_error=True,
            )
        content = safe_path.read_text(encoding="utf-8")
        return ToolResult(tool_use_id="", content=f"文件内容：\n{content}")
    except Exception as e:
        return ToolResult(tool_use_id="", content=f"读取失败：{e}", is_error=True)


def execute_write_file(filename: str, content: str) -> ToolResult:
    safe_path = WORKSPACE_DIR / Path(filename).name
    logger.debug(f"Writing file: {safe_path} ({len(content)} chars)")

    try:
        safe_path.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_use_id="",
            content=f"✓ 已写入 {filename}（{len(content)} 字符）"
        )
    except Exception as e:
        return ToolResult(tool_use_id="", content=f"写入失败：{e}", is_error=True)


def list_workspace() -> str:
    files = list(WORKSPACE_DIR.iterdir())
    return ", ".join(f.name for f in files) if files else "（空）"