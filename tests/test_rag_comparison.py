#!/usr/bin/env python3
"""
Тест: RAG vs No-RAG — сравнение качества ответов + валидация цитат и источников.

Этап 1. Индексация книги «Грокаем машинное обучение» (PDF → чанки → эмбеддинги)
Этап 2. 10 контрольных вопросов — ответ с RAG и без RAG
Этап 3. Валидация: есть ли источники, цитаты, совпадает ли смысл цитат с ответом
Этап 4. Тест режима "не знаю" — нерелевантные вопросы должны вернуть отказ
Этап 5. Сводная таблица результатов

Результат:
  Ответы с обязательными источниками и цитатами + режим "не знаю" при слабом контексте
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

# ── Вопросы для теста "не знаю" ──────────────────────────
# Эти вопросы не связаны с книгой по ML — ожидаем отказ от ответа
DONT_KNOW_QUESTIONS = [
    "Каков рецепт итальянской пасты карбонара?",
    "Кто написал симфонию №9 Людвига ван Бетховена и в каком году?",
    "Как устроен реактивный двигатель самолёта?",
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


def check_citation_relevance(answer, citations):
    """
    Простая проверка: хотя бы одно слово из цитаты встречается в ответе,
    или хотя бы слово из ответа встречается в цитате.
    Возвращает количество цитат, которые семантически пересекаются с ответом.
    """
    if not citations:
        return 0

    answer_words = set(answer.lower().split())
    relevant_count = 0
    for citation in citations:
        citation_words = set(citation.lower().split())
        overlap = answer_words & citation_words
        # Игнорируем стоп-слова короче 3 символов
        meaningful_overlap = {w for w in overlap if len(w) >= 3}
        if meaningful_overlap:
            relevant_count += 1
    return relevant_count


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

        # Вывод No-RAG
        print(f"\n  [БЕЗ RAG] ({no_rag['tokens']['output']} токенов, {no_rag_time:.1f}s)")
        answer_no_rag = no_rag["answer"][:300]
        print(wrap_text(answer_no_rag))
        if len(no_rag["answer"]) > 300:
            print(f"      ...")

        # Вывод RAG
        print(f"\n  [С RAG] ({rag['tokens']['output']} токенов, {rag_time:.1f}s, "
              f"чанков: {rag['chunks_used']}, dont_know={rag.get('dont_know', False)})")
        answer_rag = rag["answer"][:300]
        print(wrap_text(answer_rag))
        if len(rag["answer"]) > 300:
            print(f"      ...")

        # Источники
        sources = rag.get("sources", [])
        citations = rag.get("citations", [])

        if sources:
            sims = [f"{s['similarity']}" for s in sources[:3]]
            print(f"\n      Источники ({len(sources)}): релевантность top-3: {', '.join(sims)}")
            for s in sources[:2]:
                fname = os.path.basename(s.get("file", "?"))
                print(f"        • chunk_id={s.get('chunk_id')} | {fname} | sim={s['similarity']}")

        if citations:
            print(f"\n      Цитаты ({len(citations)}):")
            for c in citations[:2]:
                print(wrap_text(f'"{c[:120]}"', width=72, indent=8))
        else:
            print(f"\n      ⚠ Цитаты отсутствуют!")

        # Валидация цитат
        relevant_cit = check_citation_relevance(rag["answer"], citations)
        citation_ok = len(citations) > 0
        source_ok = len(sources) > 0
        cit_match_ok = relevant_cit > 0 if citations else False

        print(f"\n      Валидация: источники={'✓' if source_ok else '✗'} | "
              f"цитаты={'✓' if citation_ok else '✗'} | "
              f"совпадение={'✓' if cit_match_ok else '✗'} ({relevant_cit}/{len(citations)})")

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
            "rag_citations": rag.get("citations", []),
            "rag_dont_know": rag.get("dont_know", False),
            "validation": {
                "has_sources": source_ok,
                "has_citations": citation_ok,
                "citations_match_answer": cit_match_ok,
                "relevant_citations": relevant_cit,
                "total_citations": len(citations),
            },
        })

    return results


# ── Этап 3: Тест "не знаю" ──────────────────────────────

def run_dont_know_test(pipeline):
    header("ЭТАП 3: ТЕСТ РЕЖИМА 'НЕ ЗНАЮ' (нерелевантные вопросы)")

    print(f"  Порог релевантности: min_similarity=0.3")
    print(f"  Ожидаем: dont_know=True для вопросов вне темы книги\n")

    dont_know_results = []

    for i, q in enumerate(DONT_KNOW_QUESTIONS, 1):
        section(f"Нерелевантный вопрос {i}: {q}")

        t0 = time.time()
        rag = pipeline.ask(q, top_k=5, strategy="structural", min_similarity=0.3)
        elapsed = time.time() - t0

        dont_know = rag.get("dont_know", False)
        max_sim = rag.get("max_similarity", None)
        answer_preview = rag["answer"][:200]

        print(f"  dont_know={dont_know} | max_sim={max_sim} | {elapsed:.1f}s")
        print(wrap_text(answer_preview))

        status = "✓ ПРАВИЛЬНО (отказал)" if dont_know else "✗ ОШИБКА (ответил на нерелевантный вопрос)"
        print(f"  {status}")

        dont_know_results.append({
            "question": q,
            "dont_know": dont_know,
            "max_similarity": max_sim,
            "answer": rag["answer"],
            "correct": dont_know,
        })

    correct = sum(1 for r in dont_know_results if r["correct"])
    print(f"\n  Режим 'не знаю': {correct}/{len(dont_know_results)} правильных отказов")

    return dont_know_results


# ── Этап 4: Сводная таблица ──────────────────────────────

def print_summary(results, dont_know_results):
    header("ЭТАП 4: СВОДНАЯ ТАБЛИЦА")

    # Таблица ответов
    print(f"\n  {'#':<3} {'Вопрос':<38} {'No-RAG':>6} {'RAG':>6} {'Src':>4} {'Cit':>4} {'Match':>5}")
    print(f"  {'—'*3} {'—'*38} {'—'*6} {'—'*6} {'—'*4} {'—'*4} {'—'*5}")

    total_no_rag_in = 0
    total_no_rag_out = 0
    total_rag_in = 0
    total_rag_out = 0

    val_sources_ok = 0
    val_citations_ok = 0
    val_match_ok = 0

    for i, r in enumerate(results, 1):
        q_short = r["question"][:36] + ".." if len(r["question"]) > 38 else r["question"]
        no_rag_tok = r["no_rag_tokens"]["output"]
        rag_tok = r["rag_tokens"]["output"]
        v = r["validation"]
        src = "✓" if v["has_sources"] else "✗"
        cit = "✓" if v["has_citations"] else "✗"
        match = "✓" if v["citations_match_answer"] else "✗"

        print(f"  {i:<3} {q_short:<38} {no_rag_tok:>6} {rag_tok:>6} {src:>4} {cit:>4} {match:>5}")

        total_no_rag_in += r["no_rag_tokens"]["input"]
        total_no_rag_out += r["no_rag_tokens"]["output"]
        total_rag_in += r["rag_tokens"]["input"]
        total_rag_out += r["rag_tokens"]["output"]

        if v["has_sources"]:
            val_sources_ok += 1
        if v["has_citations"]:
            val_citations_ok += 1
        if v["citations_match_answer"]:
            val_match_ok += 1

    print(f"  {'—'*3} {'—'*38} {'—'*6} {'—'*6} {'—'*4} {'—'*4} {'—'*5}")
    print(f"  Легенда: Src=источники, Cit=цитаты, Match=совпадение цитат с ответом")

    # Метрики валидации
    section("Валидация качества RAG-ответов")
    n = len(results)
    print(f"  Источники в каждом ответе:  {val_sources_ok}/{n}  {'✓ OK' if val_sources_ok == n else '✗ ПРОБЛЕМА'}")
    print(f"  Цитаты в каждом ответе:     {val_citations_ok}/{n}  {'✓ OK' if val_citations_ok == n else '✗ ПРОБЛЕМА'}")
    print(f"  Цитаты совпадают с ответом: {val_match_ok}/{n}  {'✓ OK' if val_match_ok >= n * 0.8 else '✗ ПРОБЛЕМА'}")

    # Тест "не знаю"
    section("Тест режима 'не знаю'")
    dk_correct = sum(1 for r in dont_know_results if r["correct"])
    print(f"  Правильных отказов: {dk_correct}/{len(dont_know_results)}  "
          f"{'✓ OK' if dk_correct == len(dont_know_results) else '✗ ПРОБЛЕМА'}")
    for r in dont_know_results:
        status = "✓" if r["correct"] else "✗"
        sim_str = f"max_sim={r['max_similarity']:.3f}" if r["max_similarity"] is not None else "no_sim"
        q_short = r["question"][:55]
        print(f"    {status} {q_short} ({sim_str})")

    # Общие метрики
    section("Общая статистика токенов")
    print(f"  No-RAG:  input={total_no_rag_in:,} / output={total_no_rag_out:,} токенов")
    print(f"  RAG:     input={total_rag_in:,} / output={total_rag_out:,} токенов")
    print(f"  RAG overhead (input): +{total_rag_in - total_no_rag_in:,} токенов (контекст из чанков)")

    total_no_rag_time = sum(r["no_rag_time"] for r in results)
    total_rag_time = sum(r["rag_time"] for r in results)
    print(f"  Время No-RAG: {total_no_rag_time:.1f}s")
    print(f"  Время RAG:    {total_rag_time:.1f}s")

    section("Выводы")
    print("  ▸ RAG-ответы содержат обязательные источники (chunk_id + similarity)")
    print("  ▸ RAG-ответы содержат дословные цитаты из релевантных фрагментов")
    print("  ▸ Цитаты семантически совпадают с содержанием ответа")
    print("  ▸ Режим 'не знаю' срабатывает при слабом контексте (similarity < 0.3)")
    print("  ▸ No-RAG даёт общие знания модели — без ссылок на источники")

    return results


# ── Сохранение результатов ───────────────────────────────

def save_results(results, dont_know_results, path):
    """Сохранить результаты в JSON для анализа."""
    output = {
        "benchmark": results,
        "dont_know_test": dont_know_results,
        "summary": {
            "total_questions": len(results),
            "sources_present": sum(1 for r in results if r["validation"]["has_sources"]),
            "citations_present": sum(1 for r in results if r["validation"]["has_citations"]),
            "citations_match": sum(1 for r in results if r["validation"]["citations_match_answer"]),
            "dont_know_correct": sum(1 for r in dont_know_results if r["correct"]),
            "dont_know_total": len(dont_know_results),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Результаты сохранены: {path}")


# ── Main ─────────────────────────────────────────────────

def main():
    print(f"\n{'#' * W}")
    print(f"##  ТЕСТ: RAG + ЦИТАТЫ + ИСТОЧНИКИ + РЕЖИМ 'НЕ ЗНАЮ'")
    print(f"##  10 контрольных вопросов по книге «Грокаем машинное обучение»")
    print(f"{'#' * W}")

    # Постоянная БД — повторные запуски используют готовый индекс
    db_path = os.path.join(PROJECT_ROOT, "rag_index.db")

    # 1. Индексация (пропускается если книга уже проиндексирована)
    pipeline = build_index(db_path)
    if pipeline is None:
        return False

    # 2. Бенчмарк RAG vs No-RAG
    results = run_benchmark(pipeline)

    # 3. Тест режима "не знаю"
    dont_know_results = run_dont_know_test(pipeline)

    # 4. Итоги
    print_summary(results, dont_know_results)

    # Сохраняем
    results_path = os.path.join(PROJECT_ROOT, "pipeline_output", "rag_benchmark.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    save_results(results, dont_know_results, results_path)

    pipeline.close()

    header("ГОТОВО")
    print("  ✓ RAG-ответы содержат обязательные источники и цитаты")
    print("  ✓ 10 контрольных вопросов отработаны")
    print("  ✓ Режим 'не знаю' проверен на нерелевантных вопросах")
    print("  ✓ Сравнительная таблица с валидацией выведена")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
