"""SQLite base store — минимальная копия core.base_store."""

import sqlite3
from abc import ABC, abstractmethod


class BaseStore(ABC):
    """Abstract base for SQLite-backed stores."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    @abstractmethod
    def _init_tables(self):
        pass

    def close(self):
        self.conn.close()
