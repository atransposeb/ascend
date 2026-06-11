"""
auth.py – OAuth authentication, JWT management, and FastAPI dependency.

Providers configured: Google, Spotify, GitHub (via Authlib).
JWT stored in HTTP-only secure cookie.
"""

from __future__ import annotations

import uuid
import os
import httpx
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import Request, HTTPException, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from authlib.integrations.starlette_client import OAuth

from .database import get_user_by_provider_id, create_user

# Load .env from project root
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(str(ENV_PATH))
    print(f"  [auth] Loaded .env from {ENV_PATH}")
else:
    print(f"  [auth] .env not found at {ENV_PATH}")
    # Fallback: try relative to CWD
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(str(cwd_env))
        print(f"  [auth] Loaded .env from CWD: {cwd_env}")

# ── Configuration ──────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("ASCEND_JWT_SECRET", "ascend-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
COOKIE_NAME = "access_token"

# OAuth client credentials (set via environment variables for production)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

# The frontend origin for redirect URIs
BASE_URL = os.getenv("ASCEND_BASE_URL", "http://127.0.0.1:8000")

# ── OAuth client setup ─────────────────────────────────────────────────────

oauth = OAuth()

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        access_token_url="https://oauth2.googleapis.com/token",
        client_kwargs={"scope": "email profile"},
    )

if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    oauth.register(
        name="spotify",
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        authorize_url="https://accounts.spotify.com/authorize",
        access_token_url="https://accounts.spotify.com/api/token",
        client_kwargs={"scope": "user-read-email user-read-private"},
    )

if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(
        name="github",
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        authorize_url="https://github.com/login/oauth/authorize",
        access_token_url="https://github.com/login/oauth/access_token",
        client_kwargs={"scope": "user:email"},
    )


# ── JWT helpers ────────────────────────────────────────────────────────────

def create_jwt(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_jwt(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,        # set True in production with HTTPS
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        httponly=True,
        secure=False,
        samesite="lax",
    )


# ── Auth dependency ────────────────────────────────────────────────────────

def get_current_user(request: Request) -> Optional[dict]:
    """Extract the authenticated user from the JWT cookie or Authorization header.

    Returns a user dict for authenticated users, a synthetic guest dict for
    valid guest JWTs, or None for unauthenticated requests.
    """
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        return None

    payload = decode_jwt(token)
    if payload is None:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    # First try real user lookup
    user = get_user_by_provider_id(user_id)
    if user:
        return user

    # Guest users have a JWT but no DB record — return synthetic entry
    if user_id in ("guest",) or user_id.startswith("guest_"):
        return {
            "id": user_id,
            "email": "",
            "provider": "guest",
            "name": "Guest",
            "avatar_url": "",
            "created_at": "",
        }

    return None


# ── Provider helpers ───────────────────────────────────────────────────────

PROVIDER_SCOPES = {
    "google": "openid email profile",
    "spotify": "user-read-email user-read-private",
    "github": "user:email",
}

PROVIDER_NAMES = {
    "google": "Google",
    "spotify": "Spotify",
    "github": "GitHub",
}

PROVIDER_AVATAR_KEYS = {
    "google": "picture",
    "spotify": "images",
    "github": "avatar_url",
}


def extract_user_info(provider: str, userinfo: dict) -> dict:
    """Normalise user info from different OAuth providers into a standard shape."""
    uid = userinfo.get("sub") or userinfo.get("id")
    if not uid:
        uid = str(uuid.uuid4())

    email = userinfo.get("email", "")
    name = userinfo.get("name", "")

    avatar = None
    if provider == "google":
        avatar = userinfo.get("picture")
    elif provider == "spotify":
        images = userinfo.get("images", [])
        if images:
            avatar = images[0].get("url") if isinstance(images[0], dict) else images[0]
    elif provider == "github":
        avatar = userinfo.get("avatar_url")

    return {
        "provider": provider,
        "provider_user_id": str(uid),
        "email": email or "",
        "name": name or f"{provider.title()} User",
        "avatar_url": avatar or "",
    }


async def handle_oauth_callback(provider: str, request: Request) -> tuple[dict, str]:
    """Complete the OAuth flow: fetch userinfo, upsert user, return (user_dict, jwt_token)."""
    client = oauth.create_client(provider)
    token_data = await client.authorize_access_token(request)
    access_token = token_data.get("access_token")

    if provider == "google":
        async with httpx.AsyncClient() as hc:
            r = await hc.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            r.raise_for_status()
            userinfo = r.json()
    elif provider == "spotify":
        async with httpx.AsyncClient() as hc:
            r = await hc.get(
                "https://api.spotify.com/v1/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            r.raise_for_status()
            userinfo = r.json()
    elif provider == "github":
        async with httpx.AsyncClient() as hc:
            r = await hc.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            r.raise_for_status()
            userinfo = r.json()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    info = extract_user_info(provider, userinfo)
    user = create_user(
        provider=info["provider"],
        provider_user_id=info["provider_user_id"],
        email=info["email"],
        name=info["name"],
        avatar_url=info["avatar_url"],
    )
    jwt_token = create_jwt(user["id"])
    return user, jwt_token
