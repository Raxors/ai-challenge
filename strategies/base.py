from abc import ABC, abstractmethod


class ContextStrategy(ABC):

    # === Required (abstract) ===

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def prepare_messages(self, history: list) -> list:
        pass

    @abstractmethod
    def get_state_info(self) -> dict:
        pass

    # === Optional lifecycle (default no-op) ===

    def on_user_message(self, history, message):
        pass

    def on_assistant_message(self, history, message):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    # === Optional CLI hooks (default no-op) ===

    def handle_command(self, cmd, args, agent) -> bool:
        """Handle a strategy-specific command. Return True if handled."""
        return False

    def get_prompt_info(self) -> str:
        """Extra info shown in the CLI prompt (e.g. branch name)."""
        return ""

    def get_notifications(self) -> list[str]:
        """Messages to display before the prompt."""
        return []
