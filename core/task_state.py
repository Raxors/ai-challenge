"""
Task State Machine — управление жизненным циклом задачи.

TaskState отслеживает текущую фазу (planning → execution → validation → done),
текущий шаг, ожидаемое действие и поддерживает pause/resume на любом этапе.
Инжектится как системное сообщение на уровне Agent (работает со всеми стратегиями).
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


PHASES = ["planning", "execution", "validation", "done"]

TRANSITIONS = {
    "planning":   ["execution", "paused"],
    "execution":  ["validation", "planning", "paused"],
    "validation": ["done", "execution", "paused"],
    "done":       ["planning"],
    "paused":     [],  # special: resume() handles exit
}

PHASE_DESCRIPTIONS = {
    "planning":   "Planning the approach and steps",
    "execution":  "Implementing the planned steps",
    "validation": "Verifying and testing the results",
    "done":       "Task completed",
    "paused":     "Task is paused",
}

PHASE_INSTRUCTIONS = {
    "planning":   "Focus on PLANNING only. Discuss the approach, architecture, and steps. "
                  "Do NOT write implementation code yet — planning must be completed first.",
    "execution":  "Focus on IMPLEMENTATION. Follow the plan. "
                  "Do NOT skip to final results — validation is required after implementation.",
    "validation": "Focus on VERIFICATION and testing. Check the implementation against requirements. "
                  "Only after successful validation can the task be marked as done.",
    "done":       "Task is completed. Summarize results if needed.",
    "paused":     "Task is PAUSED. Do NOT repeat previous explanations when resumed.",
}


@dataclass
class TaskState:
    task_name: str = ""
    phase: str = ""
    current_step: str = ""
    expected_action: str = ""
    previous_phase: str = ""
    previous_step: str = ""
    previous_expected_action: str = ""
    created_at: str = ""
    updated_at: str = ""
    history: list = field(default_factory=list)

    # ── Persistence ─────────────────────────────────────

    def save(self, path: str) -> None:
        """Save task state to a JSON file."""
        data = asdict(self)
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str) -> "TaskState":
        """Load task state from JSON file. Returns empty state if file missing."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {
                "task_name", "phase", "current_step", "expected_action",
                "previous_phase", "previous_step", "previous_expected_action",
                "created_at", "updated_at", "history",
            }
            kwargs = {k: v for k, v in data.items() if k in known}
            return cls(**kwargs)
        except (json.JSONDecodeError, TypeError):
            return cls()

    # ── Prompt generation ───────────────────────────────

    def to_prompt(self) -> str:
        """Build a system message block with phase, step, expected action, progress bar."""
        lines = ["[Task State Machine — current task lifecycle]:"]
        lines.append(f"  Task: {self.task_name}")

        desc = PHASE_DESCRIPTIONS.get(self.phase, "")
        lines.append(f"  Current phase: {self.phase} ({desc})")

        if self.current_step:
            lines.append(f"  Current step: {self.current_step}")
        if self.expected_action:
            lines.append(f"  Expected next action: {self.expected_action}")

        # Progress bar
        progress_parts = []
        for p in PHASES:
            if p == self.phase:
                progress_parts.append(f"[{p.upper()}]")
            else:
                progress_parts.append(p)
        lines.append(f"  Progress: {' -> '.join(progress_parts)}")

        instruction = PHASE_INSTRUCTIONS.get(self.phase, f"Stay focused on the '{self.phase}' phase.")
        lines.append(f"  INSTRUCTION: {instruction}")

        valid = TRANSITIONS.get(self.phase, [])
        if valid:
            lines.append(f"  Allowed next phases: {', '.join(valid)}")

        return "\n".join(lines)

    # ── State queries ───────────────────────────────────

    def is_empty(self) -> bool:
        """True if no task is active."""
        return not self.task_name and not self.phase

    def get_valid_transitions(self) -> list:
        """Return list of phases that can be transitioned to from current phase."""
        return list(TRANSITIONS.get(self.phase, []))

    # ── State mutations ─────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_history(self) -> None:
        """Append current state to history."""
        self.history.append({
            "phase": self.phase,
            "step": self.current_step,
            "timestamp": self._now(),
        })

    def start(self, task_name: str, first_step: str = "", expected_action: str = "") -> None:
        """Begin a new task at the planning phase."""
        now = self._now()
        self.task_name = task_name
        self.phase = "planning"
        self.current_step = first_step
        self.expected_action = expected_action
        self.previous_phase = ""
        self.previous_step = ""
        self.previous_expected_action = ""
        self.created_at = now
        self.updated_at = now
        self.history = []
        self._record_history()

    def advance(self) -> str:
        """Move to the next phase in sequence. Returns new phase name.

        Raises ValueError if already at 'done' or 'paused'.
        """
        if self.phase == "paused":
            raise ValueError("Cannot advance from paused state. Use resume() first.")
        if self.phase == "done":
            raise ValueError("Already at 'done'. Use transition('planning') to restart.")

        try:
            idx = PHASES.index(self.phase)
        except ValueError:
            raise ValueError(f"Unknown phase: {self.phase}")

        new_phase = PHASES[idx + 1]
        self.phase = new_phase
        self.current_step = ""
        self.expected_action = ""
        self.updated_at = self._now()
        self._record_history()
        return new_phase

    def transition(self, target_phase: str) -> str:
        """Jump to a specific phase (validated against transition rules).

        Returns the new phase name.
        Raises ValueError if transition is not allowed.
        """
        valid = self.get_valid_transitions()
        if target_phase not in valid:
            raise ValueError(
                f"Cannot transition from '{self.phase}' to '{target_phase}'. "
                f"Valid targets: {valid}"
            )
        self.phase = target_phase
        self.current_step = ""
        self.expected_action = ""
        self.updated_at = self._now()
        self._record_history()
        return target_phase

    def pause(self) -> None:
        """Save current state and enter paused phase.

        Raises ValueError if already paused or no task active.
        """
        if self.phase == "paused":
            raise ValueError("Already paused.")
        if self.is_empty():
            raise ValueError("No active task to pause.")

        self.previous_phase = self.phase
        self.previous_step = self.current_step
        self.previous_expected_action = self.expected_action
        self.phase = "paused"
        self.updated_at = self._now()
        self._record_history()

    def resume(self) -> str:
        """Restore from paused state. Returns the restored phase name.

        Raises ValueError if not paused.
        """
        if self.phase != "paused":
            raise ValueError("Not paused. Current phase: " + self.phase)

        self.phase = self.previous_phase
        self.current_step = self.previous_step
        self.expected_action = self.previous_expected_action
        self.previous_phase = ""
        self.previous_step = ""
        self.previous_expected_action = ""
        self.updated_at = self._now()
        self._record_history()
        return self.phase

    def set_step(self, step: str, expected_action: str = "") -> None:
        """Update the current step within the current phase."""
        self.current_step = step
        self.expected_action = expected_action
        self.updated_at = self._now()

    def clear(self) -> None:
        """Reset everything to empty state."""
        self.task_name = ""
        self.phase = ""
        self.current_step = ""
        self.expected_action = ""
        self.previous_phase = ""
        self.previous_step = ""
        self.previous_expected_action = ""
        self.created_at = ""
        self.updated_at = ""
        self.history = []
