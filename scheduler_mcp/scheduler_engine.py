"""
Движок планировщика задач — SQLite + фоновый daemon-поток.

Поддерживает:
  - Одноразовые напоминания (task_type='once')
  - Периодические задачи (task_type='periodic')
  - Ручное логирование данных
  - Агрегированные сводки
"""

import sqlite3
import threading
import json
import time


class SchedulerEngine:
    """Планировщик задач с SQLite-хранилищем и фоновым потоком."""

    def __init__(self, db_path="scheduler.db", check_interval=5):
        self.db_path = db_path
        self.check_interval = check_interval
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA auto_vacuum = FULL")
        self._init_db()

    def _init_db(self):
        """Создать таблицы если не существуют."""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    callback_type TEXT NOT NULL,
                    interval_seconds INTEGER,
                    payload TEXT NOT NULL DEFAULT '{}',
                    next_run REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS task_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    delivered INTEGER DEFAULT 0,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                );
            """)
            # Миграция: добавить колонку delivered если отсутствует
            try:
                self._conn.execute("ALTER TABLE task_results ADD COLUMN delivered INTEGER DEFAULT 0")
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # колонка уже существует
            self._conn.commit()

    def start(self):
        """Запустить фоновый daemon-поток планировщика."""
        if self._thread is not None and self._thread.is_alive():
            return
        # Пометить все старые результаты как доставленные,
        # чтобы при старте не показывались накопившиеся уведомления
        with self._lock:
            self._conn.execute("UPDATE task_results SET delivered = 1 WHERE delivered = 0")
            self._conn.commit()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Остановить фоновый поток."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.check_interval + 2)
            self._thread = None

    def _run_loop(self):
        """Цикл проверки: _check_and_fire() каждые N секунд."""
        while not self._stop_event.is_set():
            try:
                self._check_and_fire()
            except Exception:
                pass  # daemon-поток не должен падать
            self._stop_event.wait(self.check_interval)

    def _check_and_fire(self):
        """Найти задачи с next_run <= now, выполнить, сохранить результат."""
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, callback_type, payload, task_type, interval_seconds "
                "FROM tasks WHERE status = 'active' AND next_run <= ?",
                (now,),
            ).fetchall()

        for row in rows:
            task_id = row["id"]
            callback_type = row["callback_type"]
            payload = row["payload"]
            task_type = row["task_type"]
            interval_seconds = row["interval_seconds"]

            result = self._execute_callback(task_id, callback_type, payload)

            with self._lock:
                self._conn.execute(
                    "INSERT INTO task_results (task_id, result) VALUES (?, ?)",
                    (task_id, json.dumps(result, ensure_ascii=False)),
                )
                if task_type == "once":
                    self._conn.execute(
                        "UPDATE tasks SET status = 'fired' WHERE id = ?",
                        (task_id,),
                    )
                else:
                    # periodic — сдвигаем next_run
                    new_next = time.time() + (interval_seconds or self.check_interval)
                    self._conn.execute(
                        "UPDATE tasks SET next_run = ? WHERE id = ?",
                        (new_next, task_id),
                    )
                    # Удаляем старые доставленные результаты, оставляя последние 100
                    self._conn.execute(
                        "DELETE FROM task_results WHERE task_id = ? AND delivered = 1 "
                        "AND id NOT IN ("
                        "  SELECT id FROM task_results WHERE task_id = ? "
                        "  ORDER BY id DESC LIMIT 100"
                        ")",
                        (task_id, task_id),
                    )
                self._conn.commit()

    def _execute_callback(self, task_id, callback_type, payload_str):
        """Генерирует результат в зависимости от типа callback."""
        try:
            payload = json.loads(payload_str)
        except (json.JSONDecodeError, TypeError):
            payload = {}

        if callback_type == "reminder":
            # Собираем сводку по активным задачам для контекста
            with self._lock:
                active_tasks = self._conn.execute(
                    "SELECT id, name, task_type, callback_type, interval_seconds, status "
                    "FROM tasks WHERE status = 'active' ORDER BY id"
                ).fetchall()
                total_tasks = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM tasks"
                ).fetchone()["cnt"]
                fired_count = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM tasks WHERE status = 'fired'"
                ).fetchone()["cnt"]
            tasks_summary = []
            for t in active_tasks:
                info = {"id": t["id"], "name": t["name"],
                        "type": t["task_type"], "callback": t["callback_type"]}
                if t["interval_seconds"]:
                    info["interval_seconds"] = t["interval_seconds"]
                tasks_summary.append(info)
            return {
                "type": "reminder",
                "task_id": task_id,
                "message": payload.get("message", "Reminder!"),
                "tasks_summary": tasks_summary,
                "stats": {
                    "total": total_tasks,
                    "active": len(active_tasks),
                    "fired": fired_count,
                },
                "fired_at": time.time(),
            }
        elif callback_type == "summary":
            # Агрегируем данные из task_results
            with self._lock:
                rows = self._conn.execute(
                    "SELECT result FROM task_results WHERE task_id = ? ORDER BY executed_at DESC LIMIT 10",
                    (task_id,),
                ).fetchall()
            data_points = []
            for r in rows:
                try:
                    data_points.append(json.loads(r["result"]))
                except (json.JSONDecodeError, TypeError):
                    pass
            return {
                "type": "summary",
                "task_id": task_id,
                "data_points": len(data_points),
                "latest": data_points[:3] if data_points else [],
                "generated_at": time.time(),
            }
        else:
            return {
                "type": callback_type,
                "task_id": task_id,
                "payload": payload,
                "fired_at": time.time(),
            }

    def add_reminder(self, name, message, delay_seconds):
        """Создать одноразовое напоминание."""
        next_run = time.time() + delay_seconds
        payload = json.dumps({"message": message}, ensure_ascii=False)
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO tasks (name, task_type, callback_type, interval_seconds, payload, next_run) "
                "VALUES (?, 'once', 'reminder', NULL, ?, ?)",
                (name, payload, next_run),
            )
            self._conn.commit()
            task_id = cursor.lastrowid
        return self._task_to_dict(task_id)

    def add_periodic(self, name, interval_seconds, task_type, payload=None):
        """Создать периодическую задачу."""
        next_run = time.time() + interval_seconds
        payload_str = json.dumps(payload or {}, ensure_ascii=False)
        callback_type = task_type  # 'summary', 'reminder', etc.
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO tasks (name, task_type, callback_type, interval_seconds, payload, next_run) "
                "VALUES (?, 'periodic', ?, ?, ?, ?)",
                (name, callback_type, interval_seconds, payload_str, next_run),
            )
            self._conn.commit()
            task_id = cursor.lastrowid
        return self._task_to_dict(task_id)

    def list_tasks(self):
        """Список всех задач."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, task_type, callback_type, interval_seconds, "
                "payload, next_run, status, created_at FROM tasks ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_task(self, task_id):
        """Детали задачи + execution_count."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, task_type, callback_type, interval_seconds, "
                "payload, next_run, status, created_at FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if not row:
                return None
            count = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM task_results WHERE task_id = ?",
                (task_id,),
            ).fetchone()["cnt"]
        result = dict(row)
        result["execution_count"] = count
        return result

    def cancel_task(self, task_id):
        """Отменить задачу."""
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = 'cancelled' WHERE id = ?",
                (task_id,),
            )
            # Пометить уже сработавшие, но недоставленные результаты как доставленные,
            # чтобы после отмены не приходило «последнее» уведомление
            self._conn.execute(
                "UPDATE task_results SET delivered = 1 WHERE task_id = ? AND delivered = 0",
                (task_id,),
            )
            self._conn.commit()
        return {"success": True}

    def cancel_all(self):
        """Отменить все активные задачи."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE tasks SET status = 'cancelled' WHERE status = 'active'"
            )
            cancelled_count = cursor.rowcount
            self._conn.execute(
                "UPDATE task_results SET delivered = 1 WHERE delivered = 0"
            )
            self._conn.commit()
        return {"success": True, "cancelled_count": cancelled_count}

    def delete_all(self):
        """Удалить все задачи и результаты из БД."""
        with self._lock:
            tasks_count = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM tasks"
            ).fetchone()["cnt"]
            self._conn.execute("DELETE FROM task_results")
            self._conn.execute("DELETE FROM tasks")
            self._conn.commit()
        return {"success": True, "deleted_count": tasks_count}

    def get_results(self, task_id, limit=20):
        """История результатов задачи."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, task_id, result, executed_at FROM task_results "
                "WHERE task_id = ? ORDER BY executed_at DESC LIMIT ?",
                (task_id, limit),
            ).fetchall()
        results = []
        for r in rows:
            entry = dict(r)
            try:
                entry["result"] = json.loads(entry["result"])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(entry)
        return results

    def log_data(self, task_id, data):
        """Вручную добавить точку данных для задачи."""
        result_str = json.dumps(data, ensure_ascii=False)
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO task_results (task_id, result, delivered) VALUES (?, ?, 1)",
                (task_id, result_str),
            )
            self._conn.commit()
            result_id = cursor.lastrowid
        return {"id": result_id, "task_id": task_id, "data": data}

    def get_summary(self):
        """Общая сводка: счётчики + последние результаты."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM tasks"
            ).fetchone()["cnt"]
            active = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM tasks WHERE status = 'active'"
            ).fetchone()["cnt"]
            fired = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM tasks WHERE status = 'fired'"
            ).fetchone()["cnt"]
            cancelled = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM tasks WHERE status = 'cancelled'"
            ).fetchone()["cnt"]
            recent = self._conn.execute(
                "SELECT id, task_id, result, executed_at FROM task_results "
                "ORDER BY executed_at DESC LIMIT 5"
            ).fetchall()
        recent_list = []
        for r in recent:
            entry = dict(r)
            try:
                entry["result"] = json.loads(entry["result"])
            except (json.JSONDecodeError, TypeError):
                pass
            recent_list.append(entry)
        return {
            "total": total,
            "active": active,
            "fired": fired,
            "cancelled": cancelled,
            "recent_results": recent_list,
        }

    def get_pending_notifications(self):
        """Вернуть недоставленные результаты и пометить их delivered=1."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tr.id, tr.task_id, tr.result, tr.executed_at, t.name as task_name "
                "FROM task_results tr JOIN tasks t ON tr.task_id = t.id "
                "WHERE tr.delivered = 0 ORDER BY tr.executed_at"
            ).fetchall()
            if rows:
                ids = [r["id"] for r in rows]
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"UPDATE task_results SET delivered = 1 WHERE id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()
        notifications = []
        for r in rows:
            entry = dict(r)
            try:
                entry["result"] = json.loads(entry["result"])
            except (json.JSONDecodeError, TypeError):
                pass
            notifications.append(entry)
        return notifications

    def close(self):
        """Остановить поток, закрыть БД."""
        self.stop()
        with self._lock:
            self._conn.close()

    def _task_to_dict(self, task_id):
        """Прочитать задачу по ID и вернуть как dict."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, task_type, callback_type, interval_seconds, "
                "payload, next_run, status, created_at FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row:
            return dict(row)
        return None
