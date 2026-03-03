from core import BaseStore


class HistoryStore(BaseStore):

    def __init__(self, db_path="chat_history.db"):
        super().__init__(db_path)

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def load(self):
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id"
        ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def add(self, role, content):
        self.conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, content),
        )
        self.conn.commit()

    def remove_last(self):
        self.conn.execute(
            "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages)"
        )
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()

    def count(self, role=None):
        if role:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = ?", (role,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0]
