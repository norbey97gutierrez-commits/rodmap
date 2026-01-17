import datetime as dt
from typing import Any, Dict

import jwt

from src.infrastructure.azure_setup import settings


def create_access_token(payload: Dict[str, Any]) -> str:
    now = dt.datetime.utcnow()
    exp = now + dt.timedelta(minutes=settings.JWT_EXPIRES_MINUTES)
    to_encode = {**payload, "iat": now, "exp": exp}
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(payload: Dict[str, Any]) -> str:
    now = dt.datetime.utcnow()
    exp = now + dt.timedelta(minutes=settings.JWT_REFRESH_EXPIRES_MINUTES)
    to_encode = {**payload, "iat": now, "exp": exp, "type": "refresh"}
    return jwt.encode(
        to_encode, settings.JWT_REFRESH_SECRET, algorithm=settings.JWT_ALGORITHM
    )


def decode_access_token(token: str) -> Dict[str, Any]:
    return jwt.decode(
        token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
    )


def decode_refresh_token(token: str) -> Dict[str, Any]:
    return jwt.decode(
        token, settings.JWT_REFRESH_SECRET, algorithms=[settings.JWT_ALGORITHM]
    )
