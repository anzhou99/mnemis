# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config.config import init_api_db
from api.routers import auth, chat, knowledge, memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化数据库"""
    init_api_db()
    yield
    # 应用关闭时的清理逻辑（如有需要）


app = FastAPI(
    title="Mnemis API",
    description="拥有长期记忆和私有知识库的 AI 研究助手",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(memory.router)


@app.get("/healthz")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
