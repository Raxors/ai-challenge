"""
Пайплайн индексации для ассистента разработчика:
  scan → chunk → embed → store → search → ask
"""

import os
from dev_assistant.indexing.chunkers import FixedSizeChunker, StructuralChunker
from dev_assistant.indexing.embedder import Embedder
from dev_assistant.indexing.index_store import IndexStore


SUPPORTED_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".py", ".js", ".ts",
    ".json", ".yaml", ".yml", ".rst", ".html", ".css",
}


class IndexingPipeline:
    """Полный пайплайн: сканирование → чанкинг → эмбеддинги → хранение."""

    def __init__(self, db_path="index.db", api_key=None):
        self.store = IndexStore(db_path=db_path)
        self.embedder = Embedder(api_key=api_key)
        self.chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=256, overlap_tokens=64),
            "structural": StructuralChunker(max_chunk_tokens=512),
        }

    def scan_directory(self, directory, extensions=None):
        exts = extensions or SUPPORTED_EXTENSIONS
        results = []

        for root, _dirs, files in os.walk(directory):
            parts = root.split(os.sep)
            if any(
                (p.startswith(".") and p != "." and p != "..")
                or p in ("venv", "node_modules", "__pycache__")
                for p in parts
            ):
                continue

            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in exts:
                    continue
                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    if text.strip():
                        results.append((filepath, text))
                except (OSError, UnicodeDecodeError):
                    continue

        return results

    def index_files(self, files, strategy="both", embed=True, force=False):
        file_list = []
        for item in files:
            if isinstance(item, str):
                with open(item, "r", encoding="utf-8", errors="ignore") as f:
                    file_list.append((item, f.read()))
            else:
                file_list.append(item)

        strategies = []
        if strategy in ("fixed_size", "both"):
            strategies.append("fixed_size")
        if strategy in ("structural", "both"):
            strategies.append("structural")

        already_indexed = self.store.get_indexed_files() if not force else set()

        total_chunks = 0
        total_embedded = 0
        skipped = 0

        for strat_name in strategies:
            chunker = self.chunkers[strat_name]

            for filepath, text in file_list:
                if (filepath, strat_name) in already_indexed:
                    skipped += 1
                    continue

                title = os.path.basename(filepath)
                chunks = chunker.chunk(text, source_file=filepath, title=title)

                if not chunks:
                    continue

                emb_bytes_list = None
                if embed:
                    texts = [c.text for c in chunks]
                    embeddings = self.embedder.embed_texts(texts)
                    emb_bytes_list = [Embedder.to_bytes(e) for e in embeddings]
                    total_embedded += len(embeddings)

                self.store.add_chunks_batch(chunks, emb_bytes_list)
                total_chunks += len(chunks)

        return {
            "files_processed": len(file_list),
            "files_skipped": skipped,
            "strategies": strategies,
            "total_chunks": total_chunks,
            "total_embedded": total_embedded,
        }

    def search(self, query, top_k=5, strategy=None):
        query_emb = self.embedder.embed_one(query)
        return self.store.search_similar(query_emb, top_k=top_k, strategy=strategy)

    def get_stats(self):
        return self.store.stats()

    def close(self):
        self.store.close()
