"""SQLite-хранилище для чанков и эмбеддингов."""

import json
import math
from dev_assistant.indexing.base_store import BaseStore


class IndexStore(BaseStore):
    """Хранит чанки, метаданные и эмбеддинги в SQLite."""

    def __init__(self, db_path="index.db"):
        super().__init__(db_path)

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                title TEXT DEFAULT '',
                strategy TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_offset INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                text TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
                vector BLOB NOT NULL
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy)"
        )
        self.conn.commit()

    def get_indexed_files(self):
        rows = self.conn.execute(
            "SELECT DISTINCT source_file, strategy FROM chunks"
        ).fetchall()
        return {(r[0], r[1]) for r in rows}

    def add_chunk(self, chunk, embedding_bytes=None):
        meta = chunk.metadata.copy()
        source_file = meta.pop("source_file", "")
        title = meta.pop("title", "")
        strategy = meta.pop("strategy", "")
        chunk_index = meta.pop("chunk_index", 0)
        start_offset = meta.pop("start_offset", 0)
        token_count = meta.pop("token_count", 0)

        cur = self.conn.execute(
            """INSERT INTO chunks
               (source_file, title, strategy, chunk_index, start_offset,
                token_count, text, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_file, title, strategy, chunk_index, start_offset,
             token_count, chunk.text, json.dumps(meta, ensure_ascii=False)),
        )
        chunk_id = cur.lastrowid

        if embedding_bytes is not None:
            self.conn.execute(
                "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                (chunk_id, embedding_bytes),
            )

        self.conn.commit()
        return chunk_id

    def add_chunks_batch(self, chunks, embeddings_bytes=None):
        ids = []
        for i, chunk in enumerate(chunks):
            emb = embeddings_bytes[i] if embeddings_bytes else None
            ids.append(self.add_chunk(chunk, emb))
        return ids

    def get_chunk(self, chunk_id):
        row = self.conn.execute(
            "SELECT id, source_file, title, strategy, chunk_index, "
            "start_offset, token_count, text, metadata_json FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def search_similar(self, query_embedding, top_k=5, strategy=None):
        from dev_assistant.indexing.embedder import Embedder

        query = "SELECT e.chunk_id, e.vector FROM embeddings e"
        params = []
        if strategy:
            query += " JOIN chunks c ON c.id = e.chunk_id WHERE c.strategy = ?"
            params.append(strategy)

        rows = self.conn.execute(query, params).fetchall()
        scored = []
        for chunk_id, vec_bytes in rows:
            vec = Embedder.from_bytes(vec_bytes)
            sim = self._cosine_similarity(query_embedding, vec)
            scored.append((chunk_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in scored[:top_k]:
            chunk = self.get_chunk(chunk_id)
            chunk["similarity"] = round(score, 4)
            results.append(chunk)
        return results

    def stats(self):
        rows = self.conn.execute("""
            SELECT strategy,
                   COUNT(*) as cnt,
                   SUM(token_count) as total_tokens,
                   AVG(token_count) as avg_tokens,
                   COUNT(DISTINCT source_file) as files
            FROM chunks GROUP BY strategy
        """).fetchall()
        return [
            {
                "strategy": r[0],
                "chunks": r[1],
                "total_tokens": r[2],
                "avg_tokens": round(r[3], 1) if r[3] else 0,
                "files": r[4],
            }
            for r in rows
        ]

    def clear_all(self):
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    @staticmethod
    def _row_to_dict(row):
        return {
            "id": row[0],
            "source_file": row[1],
            "title": row[2],
            "strategy": row[3],
            "chunk_index": row[4],
            "start_offset": row[5],
            "token_count": row[6],
            "text": row[7],
            "metadata": json.loads(row[8]) if row[8] else {},
        }

    @staticmethod
    def _cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
