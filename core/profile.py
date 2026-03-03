"""
Профиль пользователя — персонализация ответов ассистента.

UserProfile хранит предпочтения (стиль, формат, язык, ограничения)
и инжектится как системное сообщение перед каждым запросом к LLM.
"""

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path


@dataclass
class UserProfile:
    name: str = ""
    role: str = ""                    # e.g. "developer", "manager"
    response_language: str = ""       # e.g. "русский", "English"
    response_style: str = ""          # e.g. "formal", "concise", "detailed"
    format_preference: str = ""       # e.g. "markdown", "plain", "bullet_points"
    constraints: str = ""             # e.g. "no code examples", "ELI5"
    extra: dict = field(default_factory=dict)

    # ── Labels for to_prompt() ──────────────────────────

    _field_labels = {
        "name": "User name",
        "role": "User role",
        "response_language": "Respond in language",
        "response_style": "Response style",
        "format_preference": "Format",
        "constraints": "Constraints",
    }

    # ── Persistence ─────────────────────────────────────

    def save(self, path: str) -> None:
        """Save profile to a JSON file (human-editable)."""
        data = asdict(self)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "UserProfile":
        """Load profile from JSON file. Returns empty profile if file missing."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            kwargs = {k: v for k, v in data.items() if k in known}
            return cls(**kwargs)
        except (json.JSONDecodeError, TypeError):
            return cls()

    # ── Prompt generation ───────────────────────────────

    def to_prompt(self) -> str:
        """Build a system message block from non-empty fields."""
        lines = ["[User Profile — personalization preferences]:"]
        for f in fields(self):
            if f.name == "extra":
                continue
            value = getattr(self, f.name)
            if value:
                label = self._field_labels.get(f.name, f.name)
                lines.append(f"  {label}: {value}")
        if self.extra:
            for key, value in self.extra.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────

    def is_empty(self) -> bool:
        """True if no fields are set."""
        for f in fields(self):
            if f.name == "extra":
                continue
            if getattr(self, f.name):
                return False
        if self.extra:
            return False
        return True

    def clear(self) -> None:
        """Reset all fields to defaults."""
        for f in fields(self):
            if f.name == "extra":
                self.extra = {}
            else:
                setattr(self, f.name, "")

    def set_field(self, name: str, value: str) -> bool:
        """Set a field by name. Returns True if field exists, False otherwise."""
        known = {f.name for f in fields(self) if f.name != "extra"}
        if name in known:
            setattr(self, name, value)
            return True
        return False

    def get_settable_fields(self) -> list:
        """Return list of field names that can be set via set_field."""
        return [f.name for f in fields(self) if f.name != "extra"]
