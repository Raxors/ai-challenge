import json
from datetime import datetime
from core import BaseStore


LONG_TERM_CATEGORIES = [
    "profile",
    "preferences",
    "decisions",
    "knowledge",
    "projects",
    "contacts",
]


class MemoryStore(BaseStore):

    def __init__(self, db_path="memory.db"):
        super().__init__(db_path)

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, key)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL UNIQUE,
                data_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    # -- Long-term memory --

    def save_long_term(self, category: str, key: str, value):
        now = datetime.now().isoformat()
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        self.conn.execute(
            """INSERT INTO long_term_memory (category, key, value, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(category, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (category, key, value, now, now),
        )
        self.conn.commit()

    def get_long_term(self, category: str = None) -> list:
        if category:
            rows = self.conn.execute(
                "SELECT category, key, value FROM long_term_memory WHERE category = ? ORDER BY updated_at DESC",
                (category,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT category, key, value FROM long_term_memory ORDER BY category, updated_at DESC"
            ).fetchall()
        return [{"category": r[0], "key": r[1], "value": r[2]} for r in rows]

    def get_long_term_dict(self) -> dict:
        rows = self.conn.execute(
            "SELECT category, key, value FROM long_term_memory ORDER BY category"
        ).fetchall()
        result = {}
        for cat, key, value in rows:
            if cat not in result:
                result[cat] = {}
            result[cat][key] = value
        return result

    def delete_long_term(self, category: str = None, key: str = None):
        if category and key:
            self.conn.execute(
                "DELETE FROM long_term_memory WHERE category = ? AND key = ?",
                (category, key),
            )
        elif category:
            self.conn.execute(
                "DELETE FROM long_term_memory WHERE category = ?", (category,)
            )
        else:
            self.conn.execute("DELETE FROM long_term_memory")
        self.conn.commit()

    def count_long_term(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM long_term_memory").fetchone()
        return row[0]

    # -- Working memory --

    def save_working(self, task_id: str, data: dict):
        now = datetime.now().isoformat()
        data_json = json.dumps(data, ensure_ascii=False)
        self.conn.execute(
            """INSERT INTO working_memory (task_id, data_json, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at""",
            (task_id, data_json, now, now),
        )
        self.conn.commit()

    def load_working(self, task_id: str) -> dict:
        row = self.conn.execute(
            "SELECT data_json FROM working_memory WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return {}

    def list_tasks(self) -> list:
        rows = self.conn.execute(
            "SELECT task_id, data_json, updated_at FROM working_memory ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for task_id, data_json, updated_at in rows:
            data = json.loads(data_json)
            result.append({
                "task_id": task_id,
                "keys": list(data.keys()),
                "updated_at": updated_at,
            })
        return result

    def delete_working(self, task_id: str):
        self.conn.execute("DELETE FROM working_memory WHERE task_id = ?", (task_id,))
        self.conn.commit()

    def clear_all_working(self):
        self.conn.execute("DELETE FROM working_memory")
        self.conn.commit()
