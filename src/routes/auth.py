import logging
from typing import Any, Dict
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel

from src.adapters.local.user_repo import UserRepo
from src.infrastructure.azure_setup import settings
from src.infrastructure.security import create_access_token, create_refresh_token, decode_refresh_token
from src.routes.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


@router.get("/google/login")
async def google_login():
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": str(settings.GOOGLE_REDIRECT_URI),
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url)


@router.get("/google/callback")
async def google_callback(code: str = Query(...)):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": str(settings.GOOGLE_REDIRECT_URI),
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        token_resp = await client.post(token_url, data=data)
        if token_resp.status_code != 200:
            logger.error(f"Google token error: {token_resp.text}")
            raise HTTPException(status_code=400, detail="No se pudo obtener token Google")

        token_data = token_resp.json()
        id_tok = token_data.get("id_token")
        if not id_tok:
            raise HTTPException(status_code=400, detail="ID token faltante")

    try:
        idinfo = google_id_token.verify_oauth2_token(
            id_tok, google_requests.Request(), settings.GOOGLE_CLIENT_ID
        )
    except Exception as e:
        logger.error(f"Error validando id_token: {e}")
        raise HTTPException(status_code=401, detail="Token Google inválido")

    google_sub = idinfo.get("sub")
    email = idinfo.get("email", "")
    name = idinfo.get("name", "")
    picture = idinfo.get("picture", "")

    if not google_sub:
        raise HTTPException(status_code=400, detail="Token sin sub")

    user_repo = UserRepo()
    refresh_token = create_refresh_token({"sub": google_sub, "email": email})
    user = user_repo.upsert(google_sub, email, name, picture, refresh_token)

    access_token = create_access_token(
        {"sub": google_sub, "email": email, "name": name}
    )

    redirect_url = (
        f"{settings.FRONTEND_URL}/chat"
        f"?access_token={access_token}&refresh_token={refresh_token}"
    )
    return RedirectResponse(redirect_url)


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(body: RefreshRequest):
    try:
        payload = decode_refresh_token(body.refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Refresh token inválido")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Token incorrecto")

    user_repo = UserRepo()
    user = user_repo.get_by_refresh_token(body.refresh_token)
    if not user:
        raise HTTPException(status_code=401, detail="Refresh token no registrado")

    access_token = create_access_token(
        {"sub": user["google_sub"], "email": user.get("email", ""), "name": user.get("name", "")}
    )

    new_refresh = create_refresh_token({"sub": user["google_sub"], "email": user.get("email", "")})
    user_repo.upsert(
        user["google_sub"],
        user.get("email", ""),
        user.get("name", ""),
        user.get("picture", ""),
        new_refresh,
    )

    return AuthResponse(access_token=access_token, refresh_token=new_refresh, user=user)


@router.get("/me")
async def get_me(user=Depends(get_current_user)):
    return {"user": user}


class RevokeRequest(BaseModel):
    refresh_token: str


@router.post("/revoke")
async def revoke_refresh(body: RevokeRequest):
    user_repo = UserRepo()
    user = user_repo.get_by_refresh_token(body.refresh_token)
    if not user:
        raise HTTPException(status_code=404, detail="Refresh token no encontrado")
    user_repo.revoke_refresh_token(body.refresh_token)
    return {"status": "revoked"}
