"""Генерация эмбеддингов через OpenAI API (text-embedding-3-small)."""

import os
import struct
from openai import OpenAI


class Embedder:
    """Обёртка над OpenAI Embeddings API."""

    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    BATCH_SIZE = 100

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def embed_texts(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            resp = self.client.embeddings.create(model=self.MODEL, input=batch)
            batch_embs = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embs)
        return all_embeddings

    def embed_one(self, text):
        return self.embed_texts([text])[0]

    @staticmethod
    def to_bytes(embedding):
        return struct.pack(f"<{len(embedding)}f", *embedding)

    @staticmethod
    def from_bytes(data):
        n = len(data) // 4
        return list(struct.unpack(f"<{n}f", data))
