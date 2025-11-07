"""Database connection utilities."""
import sqlite3
from app.config import DB_PATH


def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn



