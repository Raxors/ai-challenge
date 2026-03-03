from abc import ABC, abstractmethod


class LLMError(Exception):
    pass


class LLMModel(ABC):

    @abstractmethod
    def generate(self, messages, max_tokens=1024):
        """Returns {"text": str, "input_tokens": int, "output_tokens": int, "total_tokens": int}"""
        pass
