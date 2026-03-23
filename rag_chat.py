#!/usr/bin/env python3
"""
RAG Chat — мини-чат с историей диалога, RAG-поиском и памятью задачи.

Архитектура:
  ┌──────────────────────────────────────────────────────────────┐
  │  RagChat.ask(question)                                       │
  │    1. RAG search → top-k чанков (cosine similarity)         │
  │    2. Threshold check → "не знаю" если sim < min_similarity │
  │    3. Build messages:                                        │
  │         system: base_prompt                                  │
  │         system: task_memory (goal/constraints/terms)         │
  │         system: rag_context (текущие чанки + metadata)       │
  │         history: последние HISTORY_WINDOW пар user/assistant │
  │         user: текущий вопрос                                 │
  │    4. LLM → JSON {answer, citations, sources_used}           │
  │    5. Update task memory (LLM extraction)                    │
  │    6. Persist history                                        │
  │    7. Return {answer, citations, sources, dont_know, memory} │
  └──────────────────────────────────────────────────────────────┘

Запуск:
  python3 rag_chat.py
  python3 rag_chat.py --db rag_index.db --topic "машинное обучение"

Команды в чате:
  /memory   — показать текущую память задачи
  /sources  — показать источники последнего ответа
  /history  — показать историю диалога
  /reset    — очистить историю и память
  /help     — справка
  /quit     — выйти
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from storage.history_store import HistoryStore

# ── Константы ────────────────────────────────────────────────

HISTORY_WINDOW = 12          # Последних N сообщений в контексте (6 пар)
DEFAULT_TOP_K = 5            # Чанков для RAG поиска
DEFAULT_MIN_SIM = 0.3        # Порог релевантности для режима "не знаю"
MEMORY_MODEL = "gpt-4o-mini" # Лёгкая модель для обновления памяти
CHAT_MODEL = "gpt-4o"        # Модель для генерации ответов
DEFAULT_INDEX_DB = os.path.join(_PROJECT_ROOT, "rag_index.db")
DEFAULT_CHAT_DB = os.path.join(_PROJECT_ROOT, "rag_chat_history.db")
DEFAULT_MEMORY_PATH = os.path.join(_PROJECT_ROOT, "rag_chat_memory.json")

# ── Промпты ──────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
Ты — полезный ассистент с доступом к базе знаний. Отвечаешь на вопросы пользователя,
используя ТОЛЬКО предоставленный RAG-контекст. Не выдумывай факты, которых нет в контексте.

ВАЖНО: Ответ ОБЯЗАТЕЛЬНО верни в формате JSON со следующими полями:
  "answer": развёрнутый ответ на основе контекста и истории диалога
  "citations": список из 1–3 дословных цитат из фрагментов (подтверждают ответ)
  "sources_used": список номеров фрагментов [1, 2, ...], которые ты использовал

Если в контексте нет нужной информации:
{"answer": "Не знаю — в предоставленном контексте нет информации по этому вопросу. Уточните запрос.", \
"citations": [], "sources_used": []}

Учитывай историю диалога — пользователь мог уточнять детали в предыдущих сообщениях.\
"""

MEMORY_EXTRACTION_PROMPT = """\
Ты — экстрактор памяти диалога. Проанализируй последний обмен и обнови JSON с памятью.

Текущая память:
{current_memory}

Последний обмен:
Пользователь: {user_message}
Ассистент: {assistant_answer}

Обнови следующие поля (сохрани старые данные, добавь новые):
  "goal": главная цель/задача пользователя (строка, обновляй если уточнилась)
  "clarified": список уточнений от пользователя (новые детали, предпочтения, контекст)
  "constraints": ограничения и требования (технические, временные, ресурсные)
  "terms": словарь зафиксированных терминов (ключ — термин, значение — определение)
  "topics_covered": список тем уже обсуждённых в диалоге

Верни ТОЛЬКО валидный JSON без markdown и без ```. Если изменений нет — верни текущую память.\
"""


# ── ChatTaskMemory ───────────────────────────────────────────

@dataclass
class ChatTaskMemory:
    """
    Память задачи диалога. Хранит:
      goal          — что пользователь хочет достичь
      clarified     — уточнения, детали, предпочтения
      constraints   — ограничения (технические, временные, ресурсные)
      terms         — зафиксированные термины и определения
      topics_covered — темы, уже обсуждённые в диалоге
    """
    goal: str = ""
    clarified: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    terms: dict = field(default_factory=dict)
    topics_covered: list = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.goal and not self.clarified and not self.constraints

    def to_prompt(self) -> str:
        """Форматировать как системное сообщение."""
        if self.is_empty():
            return ""
        lines = ["[Память задачи — контекст диалога]:"]
        if self.goal:
            lines.append(f"  Цель пользователя: {self.goal}")
        if self.clarified:
            lines.append(f"  Уточнения: {'; '.join(self.clarified[-5:])}")
        if self.constraints:
            lines.append(f"  Ограничения: {'; '.join(self.constraints)}")
        if self.terms:
            terms_str = ", ".join(f"{k}={v}" for k, v in list(self.terms.items())[:5])
            lines.append(f"  Термины: {terms_str}")
        if self.topics_covered:
            lines.append(f"  Уже обсудили: {', '.join(self.topics_covered[-5:])}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "ChatTaskMemory":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {"goal", "clarified", "constraints", "terms", "topics_covered"}
            kwargs = {k: v for k, v in data.items() if k in known}
            return cls(**kwargs)
        except (json.JSONDecodeError, TypeError):
            return cls()

    def reset(self) -> None:
        self.goal = ""
        self.clarified = []
        self.constraints = []
        self.terms = {}
        self.topics_covered = []


# ── RagChat ─────────────────────────────────────────────────

class RagChat:
    """
    Мини-чат с RAG + историей диалога + памятью задачи.

    Поток обработки каждого вопроса:
      search → threshold → build_context → inject_memory → LLM → update_memory → persist
    """

    def __init__(
        self,
        index_db_path=DEFAULT_INDEX_DB,
        chat_db_path=DEFAULT_CHAT_DB,
        memory_path=DEFAULT_MEMORY_PATH,
        top_k=DEFAULT_TOP_K,
        min_similarity=DEFAULT_MIN_SIM,
        topic=None,
    ):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.topic = topic or "документы"

        # История диалога (только user/assistant, без системных)
        self.store = HistoryStore(db_path=chat_db_path)
        self.history = [
            m for m in self.store.load()
            if m["role"] in ("user", "assistant")
        ]

        # Память задачи
        self.memory_path = memory_path
        self.memory = ChatTaskMemory.load(memory_path)

        # Последние источники (для команды /sources)
        self.last_sources = []
        self.last_citations = []

        # RAG индекс
        self._init_pipeline(index_db_path)

        # OpenAI клиент
        from openai import OpenAI
        self._client = OpenAI()

    def _init_pipeline(self, index_db_path: str) -> None:
        from indexing.pipeline import IndexingPipeline
        self._pipeline = IndexingPipeline(db_path=index_db_path)

    # ── RAG search ──────────────────────────────────────────

    def _search(self, query: str) -> list:
        """Семантический поиск по индексу. Возвращает список чанков с similarity."""
        query_emb = self._pipeline.embedder.embed_one(query)
        return self._pipeline.store.search_similar(
            query_emb, top_k=self.top_k, strategy="structural"
        )

    def _build_rag_context(self, chunks: list) -> tuple[str, list]:
        """
        Форматировать чанки как контекстный блок для LLM.
        Возвращает (context_block_str, sources_list).
        """
        parts = []
        sources = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_file", "unknown")
            sim = chunk.get("similarity", 0)
            text = chunk.get("text", "")
            parts.append(
                f"[Фрагмент {i}] (файл: {os.path.basename(source)}, релевантность: {sim:.4f})\n{text}"
            )
            sources.append({
                "chunk_num": i,
                "file": source,
                "title": chunk.get("title", os.path.basename(source)),
                "similarity": round(sim, 4),
                "chunk_id": chunk.get("id"),
            })
        return "\n\n---\n\n".join(parts), sources

    # ── LLM calls ───────────────────────────────────────────

    def _call_llm(self, messages: list) -> dict:
        """Вызов OpenAI с JSON-mode. Возвращает parsed dict + токены."""
        resp = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            parsed = {"answer": raw, "citations": [], "sources_used": []}
        return {
            "parsed": parsed,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }

    def _update_memory(self, user_message: str, assistant_answer: str) -> None:
        """LLM-обновление памяти задачи. Лёгкая модель, мало токенов."""
        try:
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                current_memory=json.dumps(self.memory.to_dict(), ensure_ascii=False, indent=2),
                user_message=user_message,
                assistant_answer=assistant_answer[:500],  # обрезаем чтобы сэкономить токены
            )
            resp = self._client.chat.completions.create(
                model=MEMORY_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Обнови память."},
                ],
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            # Обновляем только известные поля
            if "goal" in data and data["goal"]:
                self.memory.goal = data["goal"]
            if "clarified" in data and isinstance(data["clarified"], list):
                # Добавляем новые, без дублей
                existing = set(self.memory.clarified)
                for item in data["clarified"]:
                    if item and item not in existing:
                        self.memory.clarified.append(item)
                        existing.add(item)
            if "constraints" in data and isinstance(data["constraints"], list):
                existing = set(self.memory.constraints)
                for item in data["constraints"]:
                    if item and item not in existing:
                        self.memory.constraints.append(item)
                        existing.add(item)
            if "terms" in data and isinstance(data["terms"], dict):
                self.memory.terms.update(data["terms"])
            if "topics_covered" in data and isinstance(data["topics_covered"], list):
                existing = set(self.memory.topics_covered)
                for item in data["topics_covered"]:
                    if item and item not in existing:
                        self.memory.topics_covered.append(item)
                        existing.add(item)
            self.memory.save(self.memory_path)
        except Exception:
            pass  # Ошибки обновления памяти не должны ломать основной ответ

    # ── Main ask method ─────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Основной метод: задать вопрос и получить структурированный ответ.

        Возвращает dict:
          answer      — текст ответа
          citations   — дословные цитаты из чанков
          sources     — список источников (file, title, similarity, chunk_id)
          dont_know   — True если контекст нерелевантен
          memory      — текущая память задачи
          tokens      — {input, output}
        """
        # 1. RAG search
        chunks = self._search(question)

        # 2. Threshold check — режим "не знаю"
        if not chunks:
            return self._dont_know_response(
                reason="в индексе не найдено документов",
                question=question,
            )

        max_sim = max(c.get("similarity", 0) for c in chunks)
        if max_sim < self.min_similarity:
            return self._dont_know_response(
                reason=f"максимальная релевантность {max_sim:.3f} ниже порога {self.min_similarity}",
                question=question,
                max_similarity=max_sim,
            )

        # 3. Контекст из чанков
        context_block, sources = self._build_rag_context(chunks)

        # 4. Сборка сообщений для LLM
        messages = self._build_messages(question, context_block)

        # 5. Вызов LLM
        llm_result = self._call_llm(messages)
        parsed = llm_result["parsed"]

        answer = parsed.get("answer", "")
        citations = parsed.get("citations", [])
        sources_used_nums = parsed.get("sources_used", [])

        # Фильтруем источники — только использованные LLM
        used_sources = []
        if sources_used_nums:
            for idx in sources_used_nums:
                if isinstance(idx, int) and 1 <= idx <= len(sources):
                    used_sources.append(sources[idx - 1])
        if not used_sources:
            used_sources = sources  # fallback

        dont_know = (
            not citations and not sources_used_nums
            and "не знаю" in answer.lower()
        )

        # 6. Сохраняем в историю
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self.store.add("user", question)
        self.store.add("assistant", answer)

        # 7. Обновляем память задачи
        self._update_memory(question, answer)

        # 8. Запоминаем последние источники
        self.last_sources = used_sources
        self.last_citations = citations

        return {
            "answer": answer,
            "citations": citations,
            "sources": used_sources,
            "dont_know": dont_know,
            "memory": self.memory.to_dict(),
            "tokens": {
                "input": llm_result["input_tokens"],
                "output": llm_result["output_tokens"],
            },
        }

    def _build_messages(self, question: str, context_block: str) -> list:
        """
        Собрать список сообщений для LLM:
          [system: base] + [system: memory] + [system: rag_context]
          + [history window] + [user: question]
        """
        messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

        # Память задачи
        memory_prompt = self.memory.to_prompt()
        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})

        # RAG контекст (текущий запрос)
        rag_system = (
            f"[RAG-контекст для текущего вопроса из базы «{self.topic}»]:\n\n"
            f"{context_block}"
        )
        messages.append({"role": "system", "content": rag_system})

        # История диалога (последние HISTORY_WINDOW сообщений)
        recent = self.history[-HISTORY_WINDOW:] if len(self.history) > HISTORY_WINDOW else self.history
        messages.extend(recent)

        # Текущий вопрос
        messages.append({"role": "user", "content": question})
        return messages

    def _dont_know_response(self, reason: str, question: str, max_similarity=None) -> dict:
        """Сформировать ответ "не знаю" без вызова LLM."""
        answer = (
            f"Не знаю — {reason}. "
            "Уточните запрос или добавьте в базу более релевантные документы."
        )
        # Сохраняем в историю чтобы диалог был связным
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self.store.add("user", question)
        self.store.add("assistant", answer)

        result = {
            "answer": answer,
            "citations": [],
            "sources": [],
            "dont_know": True,
            "memory": self.memory.to_dict(),
            "tokens": {"input": 0, "output": 0},
        }
        if max_similarity is not None:
            result["max_similarity"] = max_similarity
        return result

    # ── Utility ─────────────────────────────────────────────

    def reset(self) -> None:
        """Очистить историю диалога и память задачи."""
        self.history = []
        self.store.clear()
        self.memory.reset()
        self.memory.save(self.memory_path)
        self.last_sources = []
        self.last_citations = []

    def get_history(self) -> list:
        return list(self.history)

    def close(self) -> None:
        self._pipeline.close()
        self.store.close()


# ── CLI ──────────────────────────────────────────────────────

W = 78  # ширина вывода


def _wrap(text: str, width=70, indent=4) -> str:
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(" " * indent + current)
            current = w
        else:
            current = current + " " + w if current else w
    if current:
        lines.append(" " * indent + current)
    return "\n".join(lines)


def _print_sources(sources: list, citations: list) -> None:
    if sources:
        print(f"\n  {'─' * (W - 4)}")
        print(f"  Источники ({len(sources)}):")
        for s in sources:
            fname = os.path.basename(s.get("file", "?"))
            print(f"    • chunk #{s.get('chunk_id')} | {fname} | sim={s['similarity']:.4f}")
    if citations:
        print(f"\n  Цитаты ({len(citations)}):")
        for c in citations:
            print(_wrap(f'"{c[:100]}"', width=70, indent=4))


def _print_memory(memory: ChatTaskMemory) -> None:
    print(f"\n  {'─' * (W - 4)}")
    print("  [Память задачи]")
    if memory.goal:
        print(f"  Цель:         {memory.goal}")
    if memory.clarified:
        print(f"  Уточнения:    {'; '.join(memory.clarified[-3:])}")
    if memory.constraints:
        print(f"  Ограничения:  {'; '.join(memory.constraints)}")
    if memory.terms:
        terms_str = ", ".join(f"{k}" for k in list(memory.terms.keys())[:5])
        print(f"  Термины:      {terms_str}")
    if memory.topics_covered:
        print(f"  Обсудили:     {', '.join(memory.topics_covered[-4:])}")
    if memory.is_empty():
        print("  (пусто — начните диалог)")


def main():
    parser = argparse.ArgumentParser(description="RAG Chat — чат с памятью задачи и источниками")
    parser.add_argument("--db", default=DEFAULT_INDEX_DB, help="Путь к индексной БД")
    parser.add_argument("--chat", default=DEFAULT_CHAT_DB, help="Путь к БД истории чата")
    parser.add_argument("--memory", default=DEFAULT_MEMORY_PATH, help="Путь к файлу памяти")
    parser.add_argument("--topic", default="машинное обучение", help="Тема базы знаний (для отображения)")
    parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM, help="Порог релевантности (0.0–1.0)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Число чанков для RAG поиска")
    parser.add_argument("--reset", action="store_true", help="Очистить историю и память при запуске")
    args = parser.parse_args()

    # Проверяем наличие индекса
    if not os.path.isfile(args.db):
        print(f"Ошибка: индексная БД не найдена: {args.db}")
        print("Создайте индекс через: python3 tests/test_rag_comparison.py")
        sys.exit(1)

    print(f"\n{'═' * W}")
    print(f"  RAG Chat — чат с памятью задачи и источниками")
    print(f"  База знаний: {args.topic} | БД: {os.path.basename(args.db)}")
    print(f"  Порог релевантности: {args.min_sim} | Top-K: {args.top_k}")
    print(f"{'═' * W}")
    print("  Команды: /memory  /sources  /history  /reset  /help  /quit")
    print(f"{'─' * W}")

    chat = RagChat(
        index_db_path=args.db,
        chat_db_path=args.chat,
        memory_path=args.memory,
        top_k=args.top_k,
        min_similarity=args.min_sim,
        topic=args.topic,
    )

    if args.reset:
        chat.reset()
        print("  История и память очищены.")

    # Показываем если есть история
    if chat.history:
        turns = len([m for m in chat.history if m["role"] == "user"])
        print(f"  Восстановлена история: {turns} вопросов")
        if not chat.memory.is_empty():
            print(f"  Цель из памяти: {chat.memory.goal or '(не определена)'}")
    print()

    last_result = None

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  До свидания!")
            break

        if not user_input:
            continue

        # ── Команды ──────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/quit", "/exit", "/q"):
                print("  До свидания!")
                break

            elif cmd == "/help":
                print(f"\n  {'─' * (W - 4)}")
                print("  Команды:")
                print("    /memory   — текущая память задачи")
                print("    /sources  — источники последнего ответа")
                print("    /history  — история диалога")
                print("    /reset    — очистить историю и память")
                print("    /quit     — выйти")
                print(f"  {'─' * (W - 4)}\n")

            elif cmd == "/memory":
                _print_memory(chat.memory)
                print()

            elif cmd == "/sources":
                if chat.last_sources or chat.last_citations:
                    _print_sources(chat.last_sources, chat.last_citations)
                else:
                    print("  Нет данных об источниках (задайте вопрос первым)")
                print()

            elif cmd == "/history":
                print(f"\n  {'─' * (W - 4)}")
                print(f"  История диалога ({len([m for m in chat.history if m['role'] == 'user'])} вопросов):")
                for i, m in enumerate(chat.history[-20:]):
                    role = "Вы" if m["role"] == "user" else "Бот"
                    preview = m["content"][:80].replace("\n", " ")
                    print(f"    [{role}] {preview}{'...' if len(m['content']) > 80 else ''}")
                print(f"  {'─' * (W - 4)}\n")

            elif cmd == "/reset":
                chat.reset()
                last_result = None
                print("  История диалога и память задачи очищены.\n")

            else:
                print(f"  Неизвестная команда: {cmd}. Введите /help\n")
            continue

        # ── Обработка вопроса ─────────────────────────────────
        print(f"  {'·' * 20} поиск... {'·' * 20}")
        t0 = time.time()

        try:
            result = chat.ask(user_input)
        except Exception as e:
            print(f"  Ошибка: {e}\n")
            continue

        elapsed = time.time() - t0
        last_result = result

        # ── Вывод ответа ──────────────────────────────────────
        print()
        if result["dont_know"]:
            print(f"  ⚠ {result['answer']}")
        else:
            # Ответ
            answer_lines = result["answer"].split("\n")
            for line in answer_lines:
                if line.strip():
                    print(_wrap(line, width=70, indent=2))
                else:
                    print()

            # Источники и цитаты
            _print_sources(result["sources"], result["citations"])

            # Статистика
            tok = result.get("tokens", {})
            print(f"\n  ─── {elapsed:.1f}s | "
                  f"in={tok.get('input', 0)} out={tok.get('output', 0)} токенов | "
                  f"память: {'цель зафиксирована' if chat.memory.goal else 'цель не определена'}")

        print()

    chat.close()


if __name__ == "__main__":
    main()
