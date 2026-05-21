import uuid
import sqlite3
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED
from api.config.config import API_DB_PATH
from api.dependencies import hash_password, verify_password, create_access_token
from api.models.auth import UserCreate, UserLogin, UserInfo, TokenResoponse
from api.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=TokenResoponse)
def rigester(body: UserCreate):
    """用户注册"""
    # 检查邮箱是否已被使用
    with sqlite3.connect(API_DB_PATH) as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email=?", (body.email,)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="该账号已注册")

    user_id = str(uuid.uuid4())
    with sqlite3.connect(API_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO users(id,email,hashed_password,display_name,created_at) VALUES (?,?,?,?,?)",
            (
                user_id,
                body.email,
                hash_password(body.password),
                body.display_name,
                datetime.now().isoformat(),
            ),
        )

    token = create_access_token(user_id=user_id)
    return TokenResoponse(access_token=token, expirse_in=86400)


@router.post("/login", response_model=TokenResoponse)
def login(body: UserLogin):
    """用户登录"""
    with sqlite3.connect(API_DB_PATH) as conn:
        row = conn.execute(
            "SELECT id,hashed_password FROM users WHERE email=?", (body.email,)
        ).fetchone()
        if not row or not verify_password(body.password, row[1]):
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED, detail="邮箱或密码错误"
            )
    token = create_access_token(row[0])
    return TokenResoponse(access_token=token, expirse_in=86400)


@router.get("/me", response_model=UserInfo)
def get_me(current_user: dict = Depends(get_current_user)):
    """ "用户信息"""
    return UserInfo(**current_user, tokens_used_today=0)
