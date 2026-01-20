import os
from typing import Any, Dict, Optional

import psycopg


class UserRepo:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL no configurado")
        self._init_db()

    def _init_db(self) -> None:
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        google_sub TEXT UNIQUE,
                        email TEXT,
                        name TEXT,
                        picture TEXT,
                        refresh_token TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def get_by_sub(self, google_sub: str) -> Optional[Dict[str, Any]]:
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT google_sub, email, name, picture, refresh_token
                    FROM users WHERE google_sub = %s
                    """,
                    (google_sub,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "google_sub": row[0],
                    "email": row[1],
                    "name": row[2],
                    "picture": row[3],
                    "refresh_token": row[4],
                }

    def upsert(
        self, google_sub: str, email: str, name: str, picture: str, refresh_token: str
    ) -> Dict[str, Any]:
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (google_sub, email, name, picture, refresh_token)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT(google_sub) DO UPDATE SET
                        email = EXCLUDED.email,
                        name = EXCLUDED.name,
                        picture = EXCLUDED.picture,
                        refresh_token = EXCLUDED.refresh_token,
                        last_login = CURRENT_TIMESTAMP
                    """,
                    (google_sub, email, name, picture, refresh_token),
                )
        return {
            "google_sub": google_sub,
            "email": email,
            "name": name,
            "picture": picture,
            "refresh_token": refresh_token,
        }

    def get_by_refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT google_sub, email, name, picture, refresh_token
                    FROM users WHERE refresh_token = %s
                    """,
                    (refresh_token,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "google_sub": row[0],
                    "email": row[1],
                    "name": row[2],
                    "picture": row[3],
                    "refresh_token": row[4],
                }

    def revoke_refresh_token(self, refresh_token: str) -> None:
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET refresh_token = NULL WHERE refresh_token = %s",
                    (refresh_token,),
                )
