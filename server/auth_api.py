"""
auth_api.py
-----------
Simple email/password auth with JWT.
- POST /api/auth/register
- POST /api/auth/login
- GET  /api/auth/me  (requires Bearer token)
"""

import os
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import asyncpg
import bcrypt
from jose import jwt, JWTError

from db import get_pool

log = logging.getLogger("app.auth")

auth_router = APIRouter(prefix="/api/auth")

JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AuthRequest(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return decode_token(creds.credentials)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_router.post("/register")
async def register(req: AuthRequest):
    pool: asyncpg.Pool = await get_pool()
    hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    try:
        row = await pool.fetchrow(
            "INSERT INTO users (email, password) VALUES ($1, $2) RETURNING id, email",
            req.email, hashed,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail="Email already registered")

    token = create_token(str(row["id"]), row["email"])
    log.info(f"Registered: {row['email']}")
    return {"token": token, "user_id": str(row["id"]), "email": row["email"]}


@auth_router.post("/login")
async def login(req: AuthRequest):
    pool: asyncpg.Pool = await get_pool()
    row = await pool.fetchrow("SELECT id, email, password FROM users WHERE email = $1", req.email)
    if not row or not bcrypt.checkpw(req.password.encode(), row["password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(str(row["id"]), row["email"])
    log.info(f"Login: {row['email']}")
    return {"token": token, "user_id": str(row["id"]), "email": row["email"]}


@auth_router.get("/me")
async def me(user=Depends(get_current_user)):
    return {"user_id": user["sub"], "email": user["email"]}