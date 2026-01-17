from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.adapters.local.user_repo import UserRepo
from src.infrastructure.security import decode_access_token

security = HTTPBearer()


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    token = creds.credentials
    try:
        payload = decode_access_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido")

    google_sub = payload.get("sub")
    if not google_sub:
        raise HTTPException(status_code=401, detail="Token sin sub")

    user = UserRepo().get_by_sub(google_sub)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")

    return user
