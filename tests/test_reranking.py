#!/usr/bin/env python3
"""
Тест: сравнение 3 режимов RAG.

  1. Базовый RAG         — top-5 по similarity, без фильтрации
  2. Enhanced RAG        — threshold + LLM-rerank
  3. Enhanced + rewrite  — query rewrite + threshold + LLM-rerank

10 контрольных вопросов по книге «Грокаем машинное обучение».

Результат:
  Улучшенный RAG: фильтрация/реранкинг + query rewrite + сравнение режимов
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
DB_PATH = os.path.join(PROJECT_ROOT, "rag_index.db")
PDF_FILE = os.path.join(
    PROJECT_ROOT, "docs",
    "Серрано Л. - Грокаем машинное обучение (Библиотека программиста) - 2024.pdf",
)

BENCHMARK = [
    {
        "q": "Что такое машинное обучение?",
        "expected": "Алгоритмы учатся на данных, распознавание паттернов",
    },
    {
        "q": "Чем отличается контролируемое обучение от неконтролируемого?",
        "expected": "Контролируемое — размеченные данные; неконтролируемое — без меток",
    },
    {
        "q": "Что такое переобучение и как с ним бороться?",
        "expected": "Модель запоминает данные; регуляризация, кросс-валидация",
    },
    {
        "q": "Как работает линейная регрессия?",
        "expected": "Подбор прямой, минимизация ошибки, y = mx + b",
    },
    {
        "q": "Что такое логистическая регрессия?",
        "expected": "Классификация, сигмоидная функция, вероятность класса",
    },
    {
        "q": "Какие метрики используются для оценки классификатора?",
        "expected": "Accuracy, precision, recall, F1, ROC/AUC",
    },
    {
        "q": "Как устроены деревья решений?",
        "expected": "Разбиение по признакам, энтропия/Джини, листья — предсказания",
    },
    {
        "q": "Что такое нейронные сети и из чего они состоят?",
        "expected": "Слои нейронов, веса, функции активации, обратное распространение",
    },
    {
        "q": "Как работает SVM?",
        "expected": "Максимизация зазора, опорные векторы, ядерные трюки",
    },
    {
        "q": "Что такое градиентный бустинг?",
        "expected": "Последовательное добавление слабых моделей, исправление ошибок",
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


def wrap(text, width=72, indent=6):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(" " * indent + cur)
            cur = w
        else:
            cur = cur + " " + w if cur else w
    if cur:
        lines.append(" " * indent + cur)
    return "\n".join(lines)


# ── Подготовка индекса ───────────────────────────────────

def ensure_index():
    """Создать или открыть индекс."""
    header("ПОДГОТОВКА ИНДЕКСА")

    pipeline = IndexingPipeline(db_path=DB_PATH)

    if pipeline.store.is_indexed(PDF_FILE, strategy="structural"):
        count = pipeline.store.count_chunks(strategy="structural")
        print(f"  ✓ Книга уже проиндексирована ({count} чанков)")
        return pipeline

    if not os.path.isfile(PDF_FILE):
        print(f"  ✗ PDF не найден: {PDF_FILE}")
        return None

    print(f"  Индексация...")
    text = _read_pdf(PDF_FILE)
    pipeline.index_files([(PDF_FILE, text)], strategy="structural", embed=True)
    count = pipeline.store.count_chunks(strategy="structural")
    print(f"  ✓ Проиндексировано: {count} чанков")
    return pipeline


# ── Запуск бенчмарка ─────────────────────────────────────

def run_benchmark(pipeline):
    header("СРАВНЕНИЕ 3 РЕЖИМОВ RAG (10 вопросов)")

    all_results = []

    for i, item in enumerate(BENCHMARK, 1):
        q = item["q"]
        section(f"Вопрос {i}/10: {q}")
        print(f"  Ожидание: {item['expected']}")

        row = {"question": q, "expected": item["expected"]}

        # ── Режим 1: Базовый RAG ──
        print(f"\n  [1] Базовый RAG (top-5, без фильтрации)...")
        t0 = time.time()
        r1 = pipeline.ask(q, top_k=5, strategy="structural")
        t1 = time.time() - t0

        sims = [s["similarity"] for s in r1.get("sources", [])]
        print(f"      Чанков: {r1['chunks_used']}, similarity: {sims}")
        print(wrap(r1["answer"][:200] + ("..." if len(r1["answer"]) > 200 else "")))
        row["basic"] = {
            "answer": r1["answer"],
            "chunks_used": r1["chunks_used"],
            "similarities": sims,
            "tokens": r1.get("tokens", {}),
            "time": round(t1, 2),
        }

        # ── Режим 2: Enhanced RAG (threshold + rerank) ──
        print(f"\n  [2] Enhanced RAG (threshold=0.45 + LLM-rerank)...")
        t0 = time.time()
        r2 = pipeline.ask_enhanced(
            q, top_k_initial=10, top_k_final=5,
            threshold=0.45, rewrite=False, strategy="structural",
        )
        t2 = time.time() - t0

        stats2 = r2.get("rerank_stats", {})
        sims2 = [s["similarity"] for s in r2.get("sources", [])]
        scores2 = [s.get("llm_score", "?") for s in r2.get("sources", [])]
        print(f"      Поиск: {stats2.get('initial_count', '?')} → "
              f"порог: {stats2.get('after_threshold', '?')} → "
              f"реранк: {stats2.get('after_rerank', '?')} → "
              f"финал: {stats2.get('final_count', '?')}")
        print(f"      LLM-скоры: {scores2}, similarity: {sims2}")
        print(wrap(r2["answer"][:200] + ("..." if len(r2["answer"]) > 200 else "")))
        row["enhanced"] = {
            "answer": r2["answer"],
            "chunks_used": r2["chunks_used"],
            "similarities": sims2,
            "llm_scores": scores2,
            "rerank_stats": stats2,
            "tokens": r2.get("tokens", {}),
            "time": round(t2, 2),
        }

        # ── Режим 3: Enhanced + query rewrite ──
        print(f"\n  [3] Enhanced + Query Rewrite...")
        t0 = time.time()
        r3 = pipeline.ask_enhanced(
            q, top_k_initial=10, top_k_final=5,
            threshold=0.45, rewrite=True, strategy="structural",
        )
        t3 = time.time() - t0

        stats3 = r3.get("rerank_stats", {})
        sims3 = [s["similarity"] for s in r3.get("sources", [])]
        scores3 = [s.get("llm_score", "?") for s in r3.get("sources", [])]
        rw = stats3.get("rewrite_query", "")
        print(f"      Rewrite: \"{rw[:80]}{'...' if len(rw) > 80 else ''}\"")
        print(f"      Поиск: {stats3.get('initial_count', '?')} → "
              f"порог: {stats3.get('after_threshold', '?')} → "
              f"реранк: {stats3.get('after_rerank', '?')} → "
              f"финал: {stats3.get('final_count', '?')}")
        print(f"      LLM-скоры: {scores3}, similarity: {sims3}")
        print(wrap(r3["answer"][:200] + ("..." if len(r3["answer"]) > 200 else "")))
        row["rewrite"] = {
            "answer": r3["answer"],
            "chunks_used": r3["chunks_used"],
            "similarities": sims3,
            "llm_scores": scores3,
            "rewrite_query": rw,
            "rerank_stats": stats3,
            "tokens": r3.get("tokens", {}),
            "time": round(t3, 2),
        }

        all_results.append(row)

    return all_results


# ── Сводная таблица ──────────────────────────────────────

def print_summary(results):
    header("СВОДНАЯ ТАБЛИЦА")

    print(f"\n  {'#':<3} {'Вопрос':<35} {'Basic':>7} {'Enhcd':>7} {'Rewrt':>7}  "
          f"{'B sim':>6} {'E sim':>6} {'R sim':>6}")
    print(f"  {'—'*3} {'—'*35} {'—'*7} {'—'*7} {'—'*7}  {'—'*6} {'—'*6} {'—'*6}")

    total = {"basic": {"tokens_in": 0, "tokens_out": 0, "time": 0, "chunks": 0},
             "enhanced": {"tokens_in": 0, "tokens_out": 0, "time": 0, "chunks": 0},
             "rewrite": {"tokens_in": 0, "tokens_out": 0, "time": 0, "chunks": 0}}

    for i, r in enumerate(results, 1):
        q = r["question"][:33] + ".." if len(r["question"]) > 35 else r["question"]

        b = r["basic"]
        e = r["enhanced"]
        w = r["rewrite"]

        b_out = b["tokens"].get("output", 0)
        e_out = e["tokens"].get("output", 0)
        w_out = w["tokens"].get("output", 0)

        b_sim = f"{b['similarities'][0]:.2f}" if b["similarities"] else "—"
        e_sim = f"{e['similarities'][0]:.2f}" if e["similarities"] else "—"
        w_sim = f"{w['similarities'][0]:.2f}" if w["similarities"] else "—"

        print(f"  {i:<3} {q:<35} {b_out:>7} {e_out:>7} {w_out:>7}  "
              f"{b_sim:>6} {e_sim:>6} {w_sim:>6}")

        for mode, data in [("basic", b), ("enhanced", e), ("rewrite", w)]:
            total[mode]["tokens_in"] += data["tokens"].get("input", 0)
            total[mode]["tokens_out"] += data["tokens"].get("output", 0)
            total[mode]["time"] += data["time"]
            total[mode]["chunks"] += data["chunks_used"]

    print(f"  {'—'*3} {'—'*35} {'—'*7} {'—'*7} {'—'*7}  {'—'*6} {'—'*6} {'—'*6}")

    section("Общая статистика")
    for mode, label in [("basic", "Базовый RAG"), ("enhanced", "Enhanced"), ("rewrite", "Enhanced+Rewrite")]:
        t = total[mode]
        print(f"  {label:<22} in={t['tokens_in']:>6,} out={t['tokens_out']:>5,}  "
              f"chunks={t['chunks']:>3}  time={t['time']:.1f}s")

    section("Выводы")
    print("  ▸ Базовый RAG: все top-5 чанков без фильтрации (быстро, но шумно)")
    print("  ▸ Enhanced: порог similarity отсекает нерелевантные, LLM-реранк сортирует по смыслу")
    print("  ▸ Enhanced+Rewrite: LLM переформулирует запрос → лучший семантический поиск")
    print("  ▸ Реранкинг убирает чанки с низкой LLM-оценкой (< 3) — меньше шума в контексте")

    # Средний LLM-скор
    enhanced_scores = []
    rewrite_scores = []
    for r in results:
        enhanced_scores.extend([s for s in r["enhanced"]["llm_scores"] if isinstance(s, (int, float))])
        rewrite_scores.extend([s for s in r["rewrite"]["llm_scores"] if isinstance(s, (int, float))])

    if enhanced_scores:
        print(f"\n  Средний LLM-скор Enhanced:        {sum(enhanced_scores)/len(enhanced_scores):.1f}/10")
    if rewrite_scores:
        print(f"  Средний LLM-скор Enhanced+Rewrite: {sum(rewrite_scores)/len(rewrite_scores):.1f}/10")


# ── Main ─────────────────────────────────────────────────

def main():
    print(f"\n{'#' * W}")
    print(f"##  ТЕСТ: РЕРАНКИНГ И ФИЛЬТРАЦИЯ — 3 РЕЖИМА RAG")
    print(f"##  Базовый RAG vs Enhanced vs Enhanced+Rewrite")
    print(f"{'#' * W}")

    pipeline = ensure_index()
    if pipeline is None:
        return False

    results = run_benchmark(pipeline)
    print_summary(results)

    # Сохраняем
    out_path = os.path.join(PROJECT_ROOT, "pipeline_output", "reranking_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Результаты: {out_path}")

    pipeline.close()

    header("ГОТОВО")
    print("  ✓ 3 режима RAG сравнены на 10 вопросах")
    print("  ✓ Реранкинг (threshold + LLM-rerank) + Query Rewrite")
    print()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
