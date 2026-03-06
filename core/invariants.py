"""
Инварианты проекта — жёсткие ограничения, которые ассистент ОБЯЗАН соблюдать.

ProjectInvariants хранит список правил [{id, category, rule}],
инжектится как системное сообщение с максимальным приоритетом.
При конфликте запроса с инвариантом — ассистент ОТКАЗЫВАЕТ.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProjectInvariants:
    invariants: list = field(default_factory=list)  # [{id, category, rule}]
    _next_id: int = field(default=1)

    # ── Persistence ─────────────────────────────────────

    def save(self, path: str) -> None:
        """Save invariants to a JSON file."""
        data = {
            "invariants": self.invariants,
            "_next_id": self._next_id,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ProjectInvariants":
        """Load invariants from JSON file. Returns empty if file missing or corrupted."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            obj = cls()
            obj.invariants = data.get("invariants", [])
            obj._next_id = data.get("_next_id", 1)
            return obj
        except (json.JSONDecodeError, TypeError):
            return cls()

    # ── Prompt generation ───────────────────────────────

    def to_prompt(self) -> str:
        """Build a system message block with strict refusal instructions."""
        lines = [
            "[Инварианты проекта — ОБЯЗАТЕЛЬНЫЕ ограничения]:",
            "  Ты ОБЯЗАН соблюдать ВСЕ перечисленные ниже инварианты.",
            "  Если запрос пользователя ПРОТИВОРЕЧИТ любому инварианту — ОТКАЖИ и объясни, какой инвариант будет нарушен.",
            "  НЕ предлагай решений, нарушающих инварианты, даже если пользователь настаивает.",
            "  При конфликте инвариантов с просьбой пользователя — инварианты имеют АБСОЛЮТНЫЙ приоритет.",
        ]
        for inv in self.invariants:
            lines.append(f"  [{inv['category']}] {inv['rule']}")
        return "\n".join(lines)

    # ── CRUD ────────────────────────────────────────────

    def add(self, category: str, rule: str) -> int:
        """Add an invariant, return its id. IDs are never reused."""
        inv_id = self._next_id
        self._next_id += 1
        self.invariants.append({"id": inv_id, "category": category, "rule": rule})
        return inv_id

    def remove(self, inv_id: int) -> bool:
        """Remove invariant by id. Returns True if found and removed."""
        for i, inv in enumerate(self.invariants):
            if inv["id"] == inv_id:
                self.invariants.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all invariants and reset id counter."""
        self.invariants = []
        self._next_id = 1

    def get_all(self) -> list:
        """Return a copy of the invariants list."""
        return list(self.invariants)

    def get_by_id(self, inv_id: int) -> Optional[dict]:
        """Find invariant by id. Returns dict or None."""
        for inv in self.invariants:
            if inv["id"] == inv_id:
                return inv
        return None

    # ── Helpers ──────────────────────────────────────────

    def is_empty(self) -> bool:
        """True if no invariants are set."""
        return len(self.invariants) == 0
