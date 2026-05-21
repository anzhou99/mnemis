from pydantic import BaseModel, EmailStr
from datetime import datetime


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    display_name: str = ""


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResoponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expirse_in: int = 3600


class UserInfo(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: datetime
    daily_token_budget: int = 100_00
    tokens_used_today: int = 0
