import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


class UserRepo:
    def __init__(self, db_path: str = "data/dev_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    google_sub TEXT UNIQUE,
                    email TEXT,
                    name TEXT,
                    picture TEXT,
                    refresh_token TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_login TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_by_sub(self, google_sub: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT google_sub, email, name, picture, refresh_token FROM users WHERE google_sub = ?",
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (google_sub, email, name, picture, refresh_token)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(google_sub) DO UPDATE SET
                    email = excluded.email,
                    name = excluded.name,
                    picture = excluded.picture,
                    refresh_token = excluded.refresh_token,
                    last_login = CURRENT_TIMESTAMP
                """,
                (google_sub, email, name, picture, refresh_token),
            )
            conn.commit()
        return {
            "google_sub": google_sub,
            "email": email,
            "name": name,
            "picture": picture,
            "refresh_token": refresh_token,
        }

    def get_by_refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT google_sub, email, name, picture, refresh_token
                FROM users WHERE refresh_token = ?
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET refresh_token = NULL WHERE refresh_token = ?",
                (refresh_token,),
            )
            conn.commit()
