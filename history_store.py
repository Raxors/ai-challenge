import sqlite3


class HistoryStore:
    """Сохраняет и загружает историю диалога в SQLite."""

    def __init__(self, db_path="chat_history.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
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
        """Загружает всю историю из базы."""
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id"
        ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def add(self, role, content):
        """Добавляет одно сообщение в базу."""
        self.conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, content),
        )
        self.conn.commit()

    def remove_last(self):
        """Удаляет последнее сообщение из базы."""
        self.conn.execute(
            "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages)"
        )
        self.conn.commit()

    def clear(self):
        """Удаляет всю историю."""
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()

    def count(self, role=None):
        """Возвращает количество сообщений. Если role указан — только для этой роли."""
        if role:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = ?", (role,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0]

    def close(self):
        """Закрывает соединение с базой."""
        self.conn.close()
