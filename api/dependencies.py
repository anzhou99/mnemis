import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.config.config import API_DB_PATH
from pwdlib import PasswordHash


# ── 密码哈希 ──────────────────────────────────────────────────────
password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return password_hash.verify(plain, hashed)
    except Exception as e:
        print(f"verify_password error: {e}")
        return False


# ── JWT ───────────────────────────────────────────────────────────
SECRET_KEY = "my-secret-key"  # 生产环境用环境变量
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, ALGORITHM)


def decode_token(token: str) -> str:
    """解析 JWT，返回 user_id"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("No user_id in token")
        return user_id
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效或已过期"
        )


# ── 依赖注入：获取当前用户 ────────────────────────────────────────
bearer_schema = HTTPBearer()


def get_current_user(
    credentoials: HTTPAuthorizationCredentials = Depends(bearer_schema),
) -> dict:
    """
    FastAPI 依赖注入：从请求头里解析 JWT，返回当前用户信息。
    在需要认证的路由里加上 Depends(get_current_user) 即可。
    """

    user_id = decode_token(credentoials.credentials)

    with sqlite3.connect(API_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE id=? AND is_active=1", (user_id,)).fetchone()
        print("+" * 10)
        print(row)
        print("+" * 10)

    if not row:
        raise HTTPException(status_code=401, detail="用户不存在")

    return dict(row)
