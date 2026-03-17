"""
Оркестратор пайплайна индексации:
  scan → chunk → embed → store
"""

import os
import glob
from indexing.chunkers import FixedSizeChunker, StructuralChunker
from indexing.embedder import Embedder
from indexing.index_store import IndexStore


# Поддерживаемые расширения для автоматического сканирования
SUPPORTED_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".py", ".js", ".ts",
    ".json", ".yaml", ".yml", ".rst", ".html", ".css",
    ".pdf",
}


def _read_pdf(filepath):
    """Извлечь текст из PDF через pymupdf."""
    import pymupdf
    text_parts = []
    with pymupdf.open(filepath) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n\n".join(text_parts)


class IndexingPipeline:
    """
    Полный пайплайн: сканирование файлов → чанкинг → эмбеддинги → сохранение.
    """

    def __init__(self, db_path="index.db", api_key=None):
        self.store = IndexStore(db_path=db_path)
        self.embedder = Embedder(api_key=api_key)
        self.chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=256, overlap_tokens=64),
            "structural": StructuralChunker(max_chunk_tokens=512),
        }

    def scan_directory(self, directory, extensions=None):
        """
        Сканирует директорию и возвращает список (filepath, text).
        """
        exts = extensions or SUPPORTED_EXTENSIONS
        results = []

        for root, _dirs, files in os.walk(directory):
            # Пропускаем скрытые директории и venv
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
                    if ext == ".pdf":
                        text = _read_pdf(filepath)
                    else:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    if text.strip():
                        results.append((filepath, text))
                except (OSError, UnicodeDecodeError, Exception):
                    continue

        return results

    def index_files(self, files, strategy="both", embed=True, force=False):
        """
        Проиндексировать список файлов.

        files: список (filepath, text) или список путей к файлам.
        strategy: "fixed_size", "structural", или "both".
        embed: генерировать ли эмбеддинги.
        force: если True — переиндексировать даже если файл уже в индексе.

        Возвращает dict со статистикой.
        """
        # Нормализация входных данных
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

        # Проверяем, что уже проиндексировано
        already_indexed = self.store.get_indexed_files() if not force else set()

        total_chunks = 0
        total_embedded = 0
        skipped = 0

        for strat_name in strategies:
            chunker = self.chunkers[strat_name]

            for filepath, text in file_list:
                # Пропускаем уже проиндексированные
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
            "store_stats": self.store.stats(),
        }

    def index_directory(self, directory, strategy="both", embed=True, extensions=None, force=False):
        """Сканировать директорию и проиндексировать все найденные файлы."""
        files = self.scan_directory(directory, extensions)
        return self.index_files(files, strategy=strategy, embed=embed, force=force)

    def search(self, query, top_k=5, strategy=None):
        """Семантический поиск по индексу."""
        query_emb = self.embedder.embed_one(query)
        return self.store.search_similar(query_emb, top_k=top_k, strategy=strategy)

    def ask(self, question, top_k=5, strategy=None, model="gpt-4o"):
        """
        RAG: вопрос пользователя → эмбеддинг → top-k чанков → LLM ответ с контекстом.

        1. vector = embedder(question)
        2. chunks = top_k_filter(vector)
        3. LLM(system=context_chunks, user=question)
        """
        # 1. Ищем релевантные чанки
        results = self.search(question, top_k=top_k, strategy=strategy)

        if not results:
            return {
                "answer": "В индексе не найдено релевантных документов для ответа на этот вопрос.",
                "chunks_used": 0,
                "sources": [],
            }

        # 2. Формируем контекст из чанков
        context_parts = []
        sources = []
        for i, chunk in enumerate(results, 1):
            source = chunk.get("source_file", "unknown")
            title = chunk.get("title", "")
            sim = chunk.get("similarity", 0)
            context_parts.append(
                f"[Фрагмент {i}] (файл: {source}, релевантность: {sim})\n{chunk['text']}"
            )
            sources.append({
                "file": source,
                "title": title,
                "similarity": sim,
                "chunk_id": chunk.get("id"),
            })

        context_block = "\n\n---\n\n".join(context_parts)

        # 3. Вызываем LLM
        from openai import OpenAI
        client = OpenAI()

        system_prompt = (
            "Ты — полезный ассистент. Используй ТОЛЬКО приведённый ниже контекст "
            "для ответа на вопрос пользователя. Если в контексте нет нужной информации, "
            "так и скажи — не выдумывай.\n\n"
            f"Контекст:\n\n{context_block}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=2048,
        )

        answer = resp.choices[0].message.content

        return {
            "answer": answer,
            "chunks_used": len(results),
            "sources": sources,
            "tokens": {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            },
        }

    def ask_no_rag(self, question, model="gpt-4o"):
        """
        Ответ LLM БЕЗ RAG — только собственные знания модели.
        Для сравнения с RAG-ответом.
        """
        from openai import OpenAI
        client = OpenAI()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты — полезный ассистент. Отвечай точно и по существу."},
                {"role": "user", "content": question},
            ],
            max_tokens=2048,
        )

        return {
            "answer": resp.choices[0].message.content,
            "chunks_used": 0,
            "sources": [],
            "tokens": {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            },
        }

    def get_stats(self):
        """Статистика хранилища."""
        return self.store.stats()

    def close(self):
        self.store.close()
