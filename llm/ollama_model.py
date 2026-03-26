"""
Обёртка над Ollama API для локальной генерации.
Совместима с интерфейсом LLMModel.
"""

import json
import requests
from core import LLMModel, LLMError


OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaModel(LLMModel):

    def __init__(self, model="qwen3.5:latest", base_url=OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, messages, max_tokens=1024, tools=None):
        """
        Вызов Ollama /api/chat.
        Возвращает dict совместимый с OpenAIModel:
          text, input_tokens, output_tokens, total_tokens
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
            },
        }

        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise LLMError("Ollama не запущен. Запустите: ollama serve")
        except requests.Timeout:
            raise LLMError("Ollama: таймаут запроса (300s).")
        except requests.HTTPError as e:
            raise LLMError(f"Ollama HTTP error: {e}")

        data = resp.json()
        message = data.get("message", {})
        text = message.get("content", "")

        # Ollama возвращает eval_count / prompt_eval_count
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        result = {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        # tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            result["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"]["arguments"]),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]

        return result

    def generate_raw(self, prompt, max_tokens=1024, temperature=0.3):
        """
        Простая генерация без chat-формата.
        Удобно для RAG когда нужен один промпт.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise LLMError("Ollama не запущен. Запустите: ollama serve")
        except requests.HTTPError as e:
            raise LLMError(f"Ollama HTTP error: {e}")

        data = resp.json()
        return {
            "text": data.get("response", ""),
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
