"""
auth_api.py — security-hardened version

Changes from original:
  1. JWT expiry reduced from no-expiry/7-day to 7 days (enforced)
  2. JWT algorithm explicitly set to HS256 (reject 'none' algorithm attacks)
  3. Password minimum length enforced server-side (not just client-side)
  4. JWT secret length validated at startup — refuses to start with weak secret
  5. Bcrypt rounds kept at 12 (good balance of security vs login latency)
"""

import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator

from db import get_pool

auth_router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

# ── JWT config ────────────────────────────────────────────────────────────────
JWT_SECRET    = os.environ["JWT_SECRET"]          # Will raise KeyError if missing
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 7

# Refuse to start if JWT_SECRET is too short (weak secret = game over for auth)
if len(JWT_SECRET) < 32:
    raise RuntimeError(
        f"JWT_SECRET must be at least 32 characters. "
        f"Current length: {len(JWT_SECRET)}. "
        f"Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

# ── Models ────────────────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    email: str
    password: str

    @field_validator('password')
    @classmethod
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

    @field_validator('email')
    @classmethod
    def email_basic_check(cls, v):
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email address')
        return v.lower().strip()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _create_token(user_id: str) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> str:
    """Returns user_id string, or raises HTTPException 401."""
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],   # Explicit list prevents 'none' algorithm attack
        )
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = verify_token(credentials.credentials)
    return {"sub": user_id}


# ── Routes ────────────────────────────────────────────────────────────────────
@auth_router.post("/api/auth/register")
async def register(req: AuthRequest):
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchval(
            "SELECT id FROM users WHERE email = $1", req.email
        )
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt(rounds=12))
        user_id = await conn.fetchval(
            "INSERT INTO users (email, password) VALUES ($1, $2) RETURNING id",
            req.email, hashed.decode(),
        )

    token = _create_token(user_id)
    return {"token": token, "user_id": str(user_id), "email": req.email}


@auth_router.post("/api/auth/login")
async def login(req: AuthRequest):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, password FROM users WHERE email = $1", req.email
        )

    if not row or not bcrypt.checkpw(req.password.encode(), row["password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(row["id"])
    return {"token": token, "user_id": str(row["id"]), "email": req.email}