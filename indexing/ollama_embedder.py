"""
Локальные эмбеддинги через Ollama API (nomic-embed-text, 768 dims).
Drop-in замена для Embedder (OpenAI).
"""

import struct
import requests


OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaEmbedder:
    """Обёртка над Ollama Embeddings API."""

    MODEL = "nomic-embed-text"
    DIMENSIONS = 768

    def __init__(self, model=None, base_url=OLLAMA_BASE_URL):
        self.model = model or self.MODEL
        self.base_url = base_url.rstrip("/")

    def embed_texts(self, texts):
        """
        Получить эмбеддинги для списка текстов.
        Возвращает список списков float.
        """
        embeddings = []
        for text in texts:
            emb = self.embed_one(text)
            embeddings.append(emb)
        return embeddings

    def embed_one(self, text):
        """Получить эмбеддинг одного текста."""
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding", [])
        if not embedding:
            raise RuntimeError(f"Ollama вернул пустой эмбеддинг для модели {self.model}")
        return embedding

    @staticmethod
    def to_bytes(embedding):
        """Сериализовать list[float] -> bytes (little-endian float32)."""
        return struct.pack(f"<{len(embedding)}f", *embedding)

    @staticmethod
    def from_bytes(data):
        """Десериализовать bytes -> list[float]."""
        n = len(data) // 4
        return list(struct.unpack(f"<{n}f", data))
