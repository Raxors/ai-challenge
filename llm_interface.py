from abc import ABC, abstractmethod


class LLMError(Exception):
    """Ошибка при вызове LLM."""
    pass


class LLMModel(ABC):
    """Абстрактный интерфейс для любой LLM."""

    @abstractmethod
    def generate(self, messages, max_tokens=1024):
        """Принимает список сообщений, возвращает dict с ответом.

        Возвращает:
            {"text": str, "input_tokens": int, "output_tokens": int, "total_tokens": int}

        Выбрасывает:
            LLMError — при любой ошибке API.
        """
        pass
