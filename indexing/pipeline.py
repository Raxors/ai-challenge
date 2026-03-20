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

    def ask(self, question, top_k=5, strategy=None, model="gpt-4o", min_similarity=0.3):
        """
        RAG: вопрос пользователя → эмбеддинг → top-k чанков → LLM ответ с контекстом.

        1. vector = embedder(question)
        2. chunks = top_k_filter(vector)
        3. Проверка порога min_similarity — если ниже, возвращаем "не знаю"
        4. LLM(system=context_chunks, user=question) → JSON {answer, citations, sources_used}

        Возвращает dict с полями:
          answer      — текст ответа
          citations   — список дословных цитат из чанков
          sources     — список использованных источников
          dont_know   — True если релевантность ниже порога
        """
        import json as _json

        # 1. Ищем релевантные чанки
        results = self.search(question, top_k=top_k, strategy=strategy)

        if not results:
            return {
                "answer": "Не знаю — в индексе не найдено документов. Добавьте документы и повторите запрос.",
                "chunks_used": 0,
                "sources": [],
                "citations": [],
                "dont_know": True,
            }

        # 2. Проверяем минимальный порог релевантности
        max_sim = max(c.get("similarity", 0) for c in results)
        if max_sim < min_similarity:
            return {
                "answer": (
                    f"Не знаю — максимальная релевантность найденных фрагментов "
                    f"({max_sim:.3f}) ниже порога ({min_similarity}). "
                    f"Уточните запрос или добавьте более релевантные документы."
                ),
                "chunks_used": 0,
                "sources": [],
                "citations": [],
                "dont_know": True,
                "max_similarity": max_sim,
            }

        # 3. Формируем контекст из чанков
        context_parts = []
        sources = []
        for i, chunk in enumerate(results, 1):
            source = chunk.get("source_file", "unknown")
            title = chunk.get("title", "")
            sim = chunk.get("similarity", 0)
            context_parts.append(
                f"[Фрагмент {i}] (файл: {source}, релевантность: {sim:.4f})\n{chunk['text']}"
            )
            sources.append({
                "file": source,
                "title": title,
                "similarity": round(sim, 4),
                "chunk_id": chunk.get("id"),
            })

        context_block = "\n\n---\n\n".join(context_parts)

        # 4. Вызываем LLM — требуем структурированный JSON-ответ
        from openai import OpenAI
        client = OpenAI()

        system_prompt = (
            "Ты — полезный ассистент. Используй ТОЛЬКО приведённый ниже контекст "
            "для ответа на вопрос пользователя.\n\n"
            "Ответ ОБЯЗАТЕЛЬНО верни в формате JSON со следующими полями:\n"
            '  "answer": развёрнутый ответ на вопрос на основе контекста\n'
            '  "citations": список из 1–3 коротких дословных цитат из фрагментов, '
            'подтверждающих ответ\n'
            '  "sources_used": список номеров фрагментов [1, 2, ...], которые ты использовал\n\n'
            "Если в контексте нет нужной информации — верни строго:\n"
            '{"answer": "Не знаю — в предоставленном контексте нет информации по этому вопросу. '
            'Уточните запрос.", "citations": [], "sources_used": []}\n\n'
            f"Контекст:\n\n{context_block}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        try:
            parsed = _json.loads(raw)
            answer = parsed.get("answer", raw)
            citations = parsed.get("citations", [])
            sources_used_nums = parsed.get("sources_used", [])
        except (_json.JSONDecodeError, AttributeError):
            answer = raw
            citations = []
            sources_used_nums = []

        # Фильтруем источники — только те, что реально использовал LLM
        used_sources = []
        if sources_used_nums:
            for idx in sources_used_nums:
                if isinstance(idx, int) and 1 <= idx <= len(sources):
                    used_sources.append(sources[idx - 1])
        if not used_sources:
            used_sources = sources  # fallback: все источники

        dont_know = (
            not citations and not sources_used_nums
            and "не знаю" in answer.lower()
        )

        return {
            "answer": answer,
            "chunks_used": len(results),
            "sources": used_sources,
            "citations": citations,
            "dont_know": dont_know,
            "tokens": {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            },
        }

    def ask_enhanced(self, question, top_k_initial=10, top_k_final=5,
                     threshold=0.45, rewrite=False, strategy=None, model="gpt-4o",
                     min_similarity=0.3):
        """
        Улучшенный RAG с реранкингом, фильтрацией и структурированным ответом.

        Пайплайн:
          1. (опционально) query rewrite — LLM переформулирует запрос
          2. поиск top_k_initial чанков по эмбеддингам
          3. фильтрация по порогу similarity
          4. LLM-реранкинг оставшихся чанков
          5. отсечение по LLM-скору (< 3)
          6. top_k_final → формирование контекста → LLM ответ (JSON: answer+citations+sources)

        Проверка min_similarity: если максимальная релевантность ниже порога — "не знаю".
        """
        import json as _json
        from indexing.reranker import Reranker
        reranker = Reranker()

        rerank_stats = {"rewrite_query": None, "rewrite_tokens": None}

        # 1. Query rewrite
        search_query = question
        if rewrite:
            rewritten, rw_tokens = reranker.rewrite_query(question)
            search_query = rewritten
            rerank_stats["rewrite_query"] = rewritten
            rerank_stats["rewrite_tokens"] = rw_tokens

        # 2. Широкий поиск
        results = self.search(search_query, top_k=top_k_initial, strategy=strategy)

        if not results:
            return {
                "answer": "Не знаю — в индексе не найдено документов. Добавьте документы и повторите запрос.",
                "chunks_used": 0,
                "sources": [],
                "citations": [],
                "dont_know": True,
                "rerank_stats": rerank_stats,
            }

        # 2a. Проверяем минимальный порог релевантности
        max_sim = max(c.get("similarity", 0) for c in results)
        if max_sim < min_similarity:
            return {
                "answer": (
                    f"Не знаю — максимальная релевантность найденных фрагментов "
                    f"({max_sim:.3f}) ниже порога ({min_similarity}). "
                    f"Уточните запрос или добавьте более релевантные документы."
                ),
                "chunks_used": 0,
                "sources": [],
                "citations": [],
                "dont_know": True,
                "max_similarity": max_sim,
                "rerank_stats": rerank_stats,
            }

        # 3–5. Threshold + LLM-rerank + score filter
        pipeline_result = reranker.enhanced_pipeline(
            question, results, threshold=threshold, top_k=top_k_final,
        )
        final_chunks = pipeline_result["chunks"]
        rerank_stats.update(pipeline_result["stats"])

        if not final_chunks:
            return {
                "answer": "Не знаю — после фильтрации не осталось достаточно релевантных фрагментов. Уточните запрос.",
                "chunks_used": 0,
                "sources": [],
                "citations": [],
                "dont_know": True,
                "rerank_stats": rerank_stats,
            }

        # 6. Формируем контекст и вызываем LLM
        context_parts = []
        sources = []
        for i, chunk in enumerate(final_chunks, 1):
            source = chunk.get("source_file", "unknown")
            sim = chunk.get("similarity", 0)
            llm_score = chunk.get("llm_score", "?")
            context_parts.append(
                f"[Фрагмент {i}] (релевантность: {sim:.4f}, LLM-скор: {llm_score})\n{chunk['text']}"
            )
            sources.append({
                "file": source,
                "title": chunk.get("title", ""),
                "similarity": round(sim, 4),
                "llm_score": llm_score,
                "chunk_id": chunk.get("id"),
            })

        context_block = "\n\n---\n\n".join(context_parts)

        from openai import OpenAI
        client = OpenAI()

        system_prompt = (
            "Ты — полезный ассистент. Используй ТОЛЬКО приведённый ниже контекст "
            "для ответа на вопрос пользователя.\n\n"
            "Ответ ОБЯЗАТЕЛЬНО верни в формате JSON со следующими полями:\n"
            '  "answer": развёрнутый ответ на вопрос на основе контекста\n'
            '  "citations": список из 1–3 коротких дословных цитат из фрагментов, '
            'подтверждающих ответ\n'
            '  "sources_used": список номеров фрагментов [1, 2, ...], которые ты использовал\n\n'
            "Если в контексте нет нужной информации — верни строго:\n"
            '{"answer": "Не знаю — в предоставленном контексте нет информации по этому вопросу. '
            'Уточните запрос.", "citations": [], "sources_used": []}\n\n'
            f"Контекст:\n\n{context_block}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        try:
            parsed = _json.loads(raw)
            answer = parsed.get("answer", raw)
            citations = parsed.get("citations", [])
            sources_used_nums = parsed.get("sources_used", [])
        except (_json.JSONDecodeError, AttributeError):
            answer = raw
            citations = []
            sources_used_nums = []

        # Фильтруем источники — только те, что реально использовал LLM
        used_sources = []
        if sources_used_nums:
            for idx in sources_used_nums:
                if isinstance(idx, int) and 1 <= idx <= len(sources):
                    used_sources.append(sources[idx - 1])
        if not used_sources:
            used_sources = sources  # fallback: все источники

        dont_know = (
            not citations and not sources_used_nums
            and "не знаю" in answer.lower()
        )

        return {
            "answer": answer,
            "chunks_used": len(final_chunks),
            "sources": used_sources,
            "citations": citations,
            "dont_know": dont_know,
            "tokens": {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            },
            "rerank_stats": rerank_stats,
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
