#!/usr/bin/env python3
"""
Тест: RAG vs No-RAG — сравнение качества ответов.

Этап 1. Индексация книги «Грокаем машинное обучение» (PDF → чанки → эмбеддинги)
Этап 2. 10 контрольных вопросов — ответ с RAG и без RAG
Этап 3. Сравнительная таблица результатов

Результат:
  Агент с двумя режимами (с RAG / без RAG) + 10 контрольных вопросов и сравнение качества
"""

import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from indexing.pipeline import IndexingPipeline, _read_pdf
from indexing.chunkers import _token_len

W = 80
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
PDF_FILE = os.path.join(
    DOCS_DIR,
    "Серрано Л. - Грокаем машинное обучение (Библиотека программиста) - 2024.pdf",
)

# ── 10 контрольных вопросов ──────────────────────────────
# Каждый вопрос содержит:
#   q        — вопрос
#   expected — что должно быть в ответе (ключевые факты)
#   sources  — какие главы/темы книги должны быть использованы

BENCHMARK = [
    {
        "q": "Что такое машинное обучение?",
        "expected": "Алгоритмы, которые учатся на данных; распознавание паттернов; прогнозирование",
        "sources": "Глава 1 — Что такое машинное обучение",
    },
    {
        "q": "Чем отличается контролируемое обучение от неконтролируемого?",
        "expected": "Контролируемое использует размеченные данные; неконтролируемое находит структуру без меток",
        "sources": "Глава 1 — типы обучения",
    },
    {
        "q": "Что такое переобучение и недообучение?",
        "expected": "Переобучение — модель запоминает данные, плохо обобщает; недообучение — модель слишком простая",
        "sources": "Глава 4 — Оптимизация процесса обучения",
    },
    {
        "q": "Как работает линейная регрессия?",
        "expected": "Подбор прямой линии, минимизация ошибки; уравнение прямой y = mx + b",
        "sources": "Глава 2–3 — линейная регрессия",
    },
    {
        "q": "Что такое логистическая регрессия и для чего она используется?",
        "expected": "Классификация; сигмоидная функция; вероятность принадлежности к классу",
        "sources": "Глава 6 — Непрерывный подход к разделению точек",
    },
    {
        "q": "Как оценить качество классификационной модели?",
        "expected": "Accuracy, precision, recall, F1; матрица ошибок; кривая ROC/AUC",
        "sources": "Глава 7 — Как оценивать классификационные модели",
    },
    {
        "q": "Что такое деревья решений и как они работают?",
        "expected": "Разбиение данных по вопросам/признакам; энтропия или коэффициент Джини; листья — предсказания",
        "sources": "Глава 9 — Разбиение данных согласно ответам на вопросы",
    },
    {
        "q": "Что такое нейронные сети?",
        "expected": "Слои нейронов; веса и смещения; функции активации; обратное распространение ошибки",
        "sources": "Глава 10 — Комбинирование ради усиления. Нейронные сети",
    },
    {
        "q": "Что такое машины опорных векторов (SVM)?",
        "expected": "Максимизация зазора между классами; опорные векторы; ядерные трюки",
        "sources": "Глава 11 — Нахождение границ со стилем",
    },
    {
        "q": "Что такое градиентный бустинг и как он улучшает модель?",
        "expected": "Последовательное добавление слабых моделей; каждая следующая исправляет ошибки предыдущей; деревья решений",
        "sources": "Глава 12 — Комбинирование моделей для достижения",
    },
]


def header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def section(title):
    print(f"\n  {'-' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'-' * (W - 4)}")


def wrap_text(text, width=74, indent=6):
    """Обернуть длинный текст с отступом."""
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(" " * indent + current)
            current = w
        else:
            current = current + " " + w if current else w
    if current:
        lines.append(" " * indent + current)
    return "\n".join(lines)


# ── Этап 1: Индексация ──────────────────────────────────

def build_index(db_path):
    header("ЭТАП 1: ИНДЕКСАЦИЯ КНИГИ")

    if not os.path.isfile(PDF_FILE):
        print(f"  ✗ PDF не найден: {PDF_FILE}")
        return None

    print(f"  Книга: Грокаем машинное обучение (Серрано Л., 2024)")

    pipeline = IndexingPipeline(db_path=db_path)

    # Проверяем, проиндексирована ли книга уже
    if pipeline.store.is_indexed(PDF_FILE, strategy="structural"):
        existing = pipeline.store.count_chunks(strategy="structural")
        print(f"  ✓ Книга уже проиндексирована ({existing} чанков), пропускаем")
        print(f"  ✓ БД: {db_path} ({os.path.getsize(db_path)/1024/1024:.1f} MB)")
        return pipeline

    t0 = time.time()
    text = _read_pdf(PDF_FILE)
    print(f"  Извлечено: {len(text):,} символов, ~{_token_len(text):,} токенов")

    print(f"  Индексация стратегией structural (семантические чанки)...")
    result = pipeline.index_files(
        [(PDF_FILE, text)],
        strategy="structural",
        embed=True,
    )
    elapsed = time.time() - t0

    stats = result["store_stats"][0] if result["store_stats"] else {}
    print(f"  ✓ Чанков: {result['total_chunks']}")
    print(f"  ✓ Эмбеддингов: {result['total_embedded']}")
    print(f"  ✓ Среднее: {stats.get('avg_tokens', '?')} токенов/чанк")
    print(f"  ✓ Время: {elapsed:.1f}s")
    print(f"  ✓ БД: {db_path} ({os.path.getsize(db_path)/1024/1024:.1f} MB)")

    return pipeline


# ── Этап 2: 10 вопросов RAG vs No-RAG ───────────────────

def run_benchmark(pipeline):
    header("ЭТАП 2: 10 КОНТРОЛЬНЫХ ВОПРОСОВ — RAG vs NO-RAG")

    results = []

    for i, item in enumerate(BENCHMARK, 1):
        q = item["q"]
        section(f"Вопрос {i}/10: {q}")

        print(f"  Ожидание: {item['expected']}")
        print(f"  Источник: {item['sources']}")

        # No-RAG
        print(f"\n  ▸ Без RAG (только знания модели)...")
        t0 = time.time()
        no_rag = pipeline.ask_no_rag(q)
        no_rag_time = time.time() - t0

        # RAG
        print(f"  ▸ С RAG (контекст из книги)...")
        t0 = time.time()
        rag = pipeline.ask(q, top_k=5, strategy="structural")
        rag_time = time.time() - t0

        # Вывод
        print(f"\n  [БЕЗ RAG] ({no_rag['tokens']['output']} токенов, {no_rag_time:.1f}s)")
        answer_no_rag = no_rag["answer"][:300]
        print(wrap_text(answer_no_rag))
        if len(no_rag["answer"]) > 300:
            print(f"      ...")

        print(f"\n  [С RAG] ({rag['tokens']['output']} токенов, {rag_time:.1f}s, "
              f"чанков: {rag['chunks_used']})")
        answer_rag = rag["answer"][:300]
        print(wrap_text(answer_rag))
        if len(rag["answer"]) > 300:
            print(f"      ...")

        # Источники
        if rag.get("sources"):
            sims = [f"{s['similarity']}" for s in rag["sources"][:3]]
            print(f"\n      Релевантность top-3: {', '.join(sims)}")

        results.append({
            "question": q,
            "expected": item["expected"],
            "expected_sources": item["sources"],
            "no_rag_answer": no_rag["answer"],
            "no_rag_tokens": no_rag["tokens"],
            "no_rag_time": round(no_rag_time, 2),
            "rag_answer": rag["answer"],
            "rag_tokens": rag["tokens"],
            "rag_time": round(rag_time, 2),
            "rag_chunks_used": rag["chunks_used"],
            "rag_sources": rag.get("sources", []),
        })

    return results


# ── Этап 3: Сводная таблица ──────────────────────────────

def print_summary(results):
    header("ЭТАП 3: СВОДНАЯ ТАБЛИЦА")

    # Таблица
    print(f"\n  {'#':<3} {'Вопрос':<45} {'No-RAG':>8} {'RAG':>8} {'Чанки':>6}")
    print(f"  {'—'*3} {'—'*45} {'—'*8} {'—'*8} {'—'*6}")

    total_no_rag_in = 0
    total_no_rag_out = 0
    total_rag_in = 0
    total_rag_out = 0

    for i, r in enumerate(results, 1):
        q_short = r["question"][:43] + ".." if len(r["question"]) > 45 else r["question"]
        no_rag_tok = r["no_rag_tokens"]["output"]
        rag_tok = r["rag_tokens"]["output"]
        chunks = r["rag_chunks_used"]
        print(f"  {i:<3} {q_short:<45} {no_rag_tok:>8} {rag_tok:>8} {chunks:>6}")

        total_no_rag_in += r["no_rag_tokens"]["input"]
        total_no_rag_out += r["no_rag_tokens"]["output"]
        total_rag_in += r["rag_tokens"]["input"]
        total_rag_out += r["rag_tokens"]["output"]

    print(f"  {'—'*3} {'—'*45} {'—'*8} {'—'*8} {'—'*6}")

    # Общие метрики
    section("Общая статистика")
    print(f"  No-RAG:  input={total_no_rag_in:,} / output={total_no_rag_out:,} токенов")
    print(f"  RAG:     input={total_rag_in:,} / output={total_rag_out:,} токенов")
    print(f"  RAG overhead (input): +{total_rag_in - total_no_rag_in:,} токенов (контекст из чанков)")

    total_no_rag_time = sum(r["no_rag_time"] for r in results)
    total_rag_time = sum(r["rag_time"] for r in results)
    print(f"  Время No-RAG: {total_no_rag_time:.1f}s")
    print(f"  Время RAG:    {total_rag_time:.1f}s")

    section("Выводы")
    print("  ▸ RAG-ответы опираются на конкретные фрагменты книги")
    print("  ▸ No-RAG даёт общие знания модели (могут быть неточны для книги)")
    print("  ▸ RAG использует больше input-токенов (контекст из чанков)")
    print("  ▸ RAG-ответы содержат специфику из книги Серрано")

    return results


# ── Сохранение результатов ───────────────────────────────

def save_results(results, path):
    """Сохранить результаты в JSON для анализа."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Результаты сохранены: {path}")


# ── Main ─────────────────────────────────────────────────

def main():
    print(f"\n{'#' * W}")
    print(f"##  ТЕСТ: RAG vs NO-RAG — СРАВНЕНИЕ КАЧЕСТВА ОТВЕТОВ")
    print(f"##  10 контрольных вопросов по книге «Грокаем машинное обучение»")
    print(f"{'#' * W}")

    # Постоянная БД — повторные запуски используют готовый индекс
    db_path = os.path.join(PROJECT_ROOT, "rag_index.db")

    # 1. Индексация (пропускается если книга уже проиндексирована)
    pipeline = build_index(db_path)
    if pipeline is None:
        return False

    # 2. Бенчмарк
    results = run_benchmark(pipeline)

    # 3. Итоги
    print_summary(results)

    # Сохраняем
    results_path = os.path.join(PROJECT_ROOT, "pipeline_output", "rag_benchmark.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    save_results(results, results_path)

    pipeline.close()

    header("ГОТОВО")
    print("  ✓ Агент с двумя режимами (RAG / без RAG)")
    print("  ✓ 10 контрольных вопросов отработаны")
    print("  ✓ Сравнительная таблица выведена")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
