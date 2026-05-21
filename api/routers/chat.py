import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api.dependencies import get_current_user
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["对话"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # 不传则创建新会话


def _make_sse(event_type: str, data: dict) -> str:
    """把事件格式化为 SSE 消息"""
    payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _run_agent_streaming(
    user_id: str,
    message: str,
    session_id: str | None,
):
    """
    核心生成器：逐步 yield SSE 事件。
    实际 Agent 执行在线程池里，通过 asyncio.Queue 把事件传回协程。
    """
    import core.tools
    from core.memory.manager import MemoryEnabledAgent

    # 每个用户有自己的 Agent 实例（用户数据隔离通过 user_id 前缀实现）
    # TODO P6·2 完成后在这里加入用户级别的 DB 隔离
    agent = MemoryEnabledAgent()

    if session_id:
        # 恢复已有会话上下文（简化版，完整实现需要加载历史消息）
        pass
    else:
        agent.start_session()

    #  发送「开始处理」事件
    yield _make_sse("start", {"message": "处理中..."})

    # 在线程池执行同步 Agent（避免阻塞事件循环）
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_sync():
        try:
            result = agent.chat(message)
            loop.call_soon_threadsafe(queue.put_nowait, ("answer", result))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    # 启动同步执行（不阻塞当前协程）
    loop.run_in_executor(None, run_sync)

    # 从队列读取事件并推送给客户端
    while True:
        event_type, data = await asyncio.wait_for(queue.get(), timeout=120)
        if event_type == "answer":
            yield _make_sse("answer", {"content": data})
        elif event_type == "error":
            yield _make_sse("error", {"message": data})
            break
        elif event_type == "done":
            yield _make_sse("done", {})
            break


@router.post("/stream")
async def chat_stream(
    body: ChatRequest, current_user: dict = Depends(get_current_user)
):
    """
    流式对话接口。
    返回 text/event-stream，客户端用 EventSource 或 fetch 读取。
    """
    # 检查 token 预算
    if not _check_budget(current_user["id"]):
        raise HTTPException(
            status_code=429, detail="今日 Token 预算已用完，明天再来 🌙"
        )

    return StreamingResponse(
        _run_agent_streaming(
            user_id=current_user["id"], message=body.message, session_id=body.session_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁止 Nginx 缓冲，确保实时推送
        },
    )


def _check_budget(user_id: str) -> bool:
    """检查用户今日 token 预算是否还有剩余"""
    import sqlite3
    from datetime import date
    from api.config.config import API_DB_PATH

    today = date.today().isoformat()
    with sqlite3.connect(API_DB_PATH) as conn:
        # 获取用户预算
        budget_row = conn.execute(
            "SELECT daily_token_budget FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not budget_row:
            return False
        budget = budget_row[0]

        # 获取今日已用量
        usage_row = conn.execute(
            """SELECT COALESCE(input_tokens + output_tokens, 0)
               FROM token_usage WHERE user_id=? AND date=?""",
            (user_id, today),
        ).fetchone()
        used = usage_row[0] if usage_row else 0

    return used < budget
