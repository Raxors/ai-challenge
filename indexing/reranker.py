"""
Реранкинг и фильтрация чанков после первичного поиска.

Три режима:
  1. threshold  — отсечение по порогу similarity
  2. llm_rerank — LLM оценивает релевантность каждого чанка вопросу (0–10)
  3. combined   — threshold + llm_rerank

Плюс query rewriting — LLM переформулирует запрос для лучшего поиска.
"""

from openai import OpenAI


class Reranker:
    """Реранкинг и фильтрация результатов поиска."""

    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI()
        return self._client

    # ── Порог отсечения ──────────────────────────────────

    def filter_by_threshold(self, chunks, threshold=0.45):
        """
        Отсечь чанки с similarity ниже порога.
        Возвращает (passed, filtered_out).
        """
        passed = [c for c in chunks if c.get("similarity", 0) >= threshold]
        filtered_out = [c for c in chunks if c.get("similarity", 0) < threshold]
        return passed, filtered_out

    # ── LLM-реранкинг ───────────────────────────────────

    def llm_rerank(self, question, chunks, top_k=5):
        """
        LLM оценивает каждый чанк по релевантности к вопросу (0–10).
        Возвращает чанки отсортированные по LLM-скору.
        """
        if not chunks:
            return []

        # Формируем пакетный запрос
        chunk_texts = []
        for i, c in enumerate(chunks):
            text_preview = c["text"][:300]
            chunk_texts.append(f"[{i}] {text_preview}")

        chunks_block = "\n---\n".join(chunk_texts)

        prompt = (
            f"Вопрос пользователя: \"{question}\"\n\n"
            f"Фрагменты документа:\n{chunks_block}\n\n"
            f"Оцени релевантность КАЖДОГО фрагмента к вопросу по шкале 0–10.\n"
            f"Ответь СТРОГО в формате: одна строка на фрагмент, только номер и оценка.\n"
            f"Пример:\n0: 8\n1: 3\n2: 9\n"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты — оценщик релевантности. Отвечай только оценками."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0,
        )

        scores_text = resp.choices[0].message.content
        tokens_used = {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }

        # Парсим оценки
        scores = {}
        for line in scores_text.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":")
                try:
                    idx = int(parts[0].strip())
                    score = float(parts[1].strip())
                    scores[idx] = score
                except (ValueError, IndexError):
                    continue

        # Присваиваем оценки чанкам
        scored_chunks = []
        for i, c in enumerate(chunks):
            c_copy = c.copy()
            c_copy["llm_score"] = scores.get(i, 0)
            scored_chunks.append(c_copy)

        # Сортируем по LLM-скору
        scored_chunks.sort(key=lambda x: x["llm_score"], reverse=True)

        return scored_chunks[:top_k], tokens_used

    # ── Query rewriting ──────────────────────────────────

    def rewrite_query(self, question):
        """
        LLM переформулирует вопрос для лучшего семантического поиска.
        Возвращает (rewritten_query, tokens_used).
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — специалист по поисковым запросам. "
                        "Переформулируй вопрос пользователя так, чтобы он лучше подходил "
                        "для семантического поиска по базе знаний. "
                        "Добавь ключевые термины и синонимы. "
                        "Ответь ТОЛЬКО переформулированным запросом, без пояснений."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=150,
            temperature=0.3,
        )

        rewritten = resp.choices[0].message.content.strip()
        tokens_used = {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }

        return rewritten, tokens_used

    # ── Комбинированный пайплайн ─────────────────────────

    def enhanced_pipeline(self, question, chunks, threshold=0.45, top_k=5, rewrite=False):
        """
        Полный пайплайн: threshold → llm_rerank → top-k.

        Возвращает dict:
          chunks — финальные чанки
          stats — статистика каждого этапа
        """
        stats = {
            "initial_count": len(chunks),
            "threshold": threshold,
            "rewrite_used": rewrite,
            "rewrite_tokens": None,
            "reranker_tokens": None,
        }

        # 1. Порог
        passed, filtered_out = self.filter_by_threshold(chunks, threshold)
        stats["after_threshold"] = len(passed)
        stats["filtered_out"] = len(filtered_out)

        if not passed:
            return {"chunks": [], "stats": stats}

        # 2. LLM-реранкинг
        reranked, rerank_tokens = self.llm_rerank(question, passed, top_k=top_k)
        stats["after_rerank"] = len(reranked)
        stats["reranker_tokens"] = rerank_tokens

        # 3. Фильтруем по LLM-скору (< 3 = нерелевантно)
        final = [c for c in reranked if c.get("llm_score", 0) >= 3]
        stats["final_count"] = len(final)

        return {"chunks": final, "stats": stats}
