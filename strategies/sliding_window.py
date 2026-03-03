from core import split_messages
from strategies.base import ContextStrategy


class SlidingWindowStrategy(ContextStrategy):

    def __init__(self, window_size=10, **kwargs):
        self.window_size = window_size
        self.dropped_count = 0

    def get_name(self) -> str:
        return f"SlidingWindow(N={self.window_size})"

    def prepare_messages(self, history: list) -> list:
        system_msgs, recent = split_messages(history, self.window_size)
        non_system = [m for m in history if m["role"] != "system"]
        if len(non_system) > self.window_size:
            self.dropped_count += len(non_system) - self.window_size
        return system_msgs + recent

    def reset(self):
        self.dropped_count = 0

    def get_state_info(self) -> dict:
        return {
            "strategy": self.get_name(),
            "window_size": self.window_size,
            "dropped_messages": self.dropped_count,
        }
