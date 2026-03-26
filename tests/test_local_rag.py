#!/usr/bin/env python3
"""
Тест: Локальная RAG (Ollama) vs Облачная RAG (OpenAI)

Полностью локальный RAG-пайплайн:
  - Эмбеддинги: nomic-embed-text (Ollama, 768 dims)
  - Генерация: qwen3.5 (Ollama, ~6.6 GB)

Сравнение с облачным:
  - Эмбеддинги: text-embedding-3-small (OpenAI, 1536 dims)
  - Генерация: gpt-4o (OpenAI)

Оценка: качество, скорость, стабильность.
"""

import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from indexing.pipeline import IndexingPipeline
from indexing.ollama_embedder import OllamaEmbedder
from indexing.chunkers import _token_len

W = 80
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DOC_FILE = os.path.join(DOCS_DIR, "ml_textbook.md")

OLLAMA_MODEL = "qwen3.5:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ── 5 контрольных вопросов ────────────────────────────────
BENCHMARK = [
    {
        "q": "Что такое машинное обучение?",
        "expected": "Алгоритмы, которые учатся на данных",
    },
    {
        "q": "Чем отличается контролируемое обучение от неконтролируемого?",
        "expected": "Контролируемое — размеченные данные; неконтролируемое — без меток",
    },
    {
        "q": "Что такое переобучение и недообучение?",
        "expected": "Переобучение — модель запоминает; недообучение — слишком простая",
    },
    {
        "q": "Как работает линейная регрессия?",
        "expected": "Подбор прямой линии, минимизация ошибки, y = mx + b",
    },
    {
        "q": "Что такое нейронные сети?",
        "expected": "Слои нейронов, веса, функции активации, обратное распространение",
    },
]

DONT_KNOW_QUESTIONS = [
    "Каков рецепт итальянской пасты карбонара?",
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


def check_answer_quality(answer, expected):
    """Простая оценка: сколько ключевых слов из expected есть в answer."""
    expected_words = set(expected.lower().replace(",", "").replace(";", "").split())
    answer_words = set(answer.lower().split())
    # Только значимые слова (>3 символов)
    expected_meaningful = {w for w in expected_words if len(w) > 3}
    if not expected_meaningful:
        return 0.0
    overlap = expected_meaningful & answer_words
    return len(overlap) / len(expected_meaningful)


# ── Этап 1: Индексация с локальными эмбеддингами ──────────

def build_local_index(db_path):
    header("ЭТАП 1: ИНДЕКСАЦИЯ (локальные эмбеддинги — nomic-embed-text)")

    if not os.path.isfile(DOC_FILE):
        print(f"  ✗ Документ не найден: {DOC_FILE}")
        return None

    print(f"  Документ: {os.path.basename(DOC_FILE)}")
    print(f"  Embedder: {OLLAMA_EMBED_MODEL} (Ollama, 768 dims)")

    embedder = OllamaEmbedder(model=OLLAMA_EMBED_MODEL)
    pipeline = IndexingPipeline(db_path=db_path, embedder=embedder)

    if pipeline.store.is_indexed(DOC_FILE, strategy="structural"):
        existing = pipeline.store.count_chunks(strategy="structural")
        print(f"  ✓ Уже проиндексирована ({existing} чанков), пропускаем")
        return pipeline

    t0 = time.time()
    with open(DOC_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  Извлечено: {len(text):,} символов, ~{_token_len(text):,} токенов")

    print(f"  Индексация стратегией structural...")
    result = pipeline.index_files(
        [(DOC_FILE, text)],
        strategy="structural",
        embed=True,
    )
    elapsed = time.time() - t0

    stats = result["store_stats"][0] if result["store_stats"] else {}
    print(f"  ✓ Чанков: {result['total_chunks']}")
    print(f"  ✓ Эмбеддингов: {result['total_embedded']}")
    print(f"  ✓ Время индексации: {elapsed:.1f}s")
    print(f"  ✓ БД: {db_path} ({os.path.getsize(db_path)/1024/1024:.1f} MB)")

    return pipeline


# ── Этап 2: Сравнение Local vs Cloud ──────────────────────

def run_comparison(local_pipeline, cloud_pipeline):
    header("ЭТАП 2: СРАВНЕНИЕ — ЛОКАЛЬНАЯ vs ОБЛАЧНАЯ RAG")

    results = []

    for i, item in enumerate(BENCHMARK, 1):
        q = item["q"]
        section(f"Вопрос {i}/{len(BENCHMARK)}: {q}")
        print(f"  Ожидание: {item['expected']}")

        row = {"question": q, "expected": item["expected"]}

        # ── Локальная RAG (Ollama) ──
        print(f"\n  ▸ Локальная RAG (Ollama {OLLAMA_MODEL})...")
        t0 = time.time()
        local_rag = local_pipeline.ask_local(
            q, top_k=5, strategy="structural", ollama_model=OLLAMA_MODEL
        )
        local_time = time.time() - t0

        answer_local = local_rag["answer"][:400]
        print(f"    [{local_time:.1f}s, {local_rag['tokens']['output']} tok, "
              f"chunks={local_rag['chunks_used']}]")
        print(wrap_text(answer_local, indent=8))

        row["local_rag"] = {
            "answer": local_rag["answer"],
            "time": round(local_time, 2),
            "tokens": local_rag["tokens"],
            "chunks_used": local_rag["chunks_used"],
            "citations": local_rag.get("citations", []),
            "sources": local_rag.get("sources", []),
            "dont_know": local_rag.get("dont_know", False),
        }

        # ── Облачная RAG (OpenAI) ──
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        if has_openai and cloud_pipeline:
            print(f"\n  ▸ Облачная RAG (OpenAI gpt-4o)...")
            t0 = time.time()
            cloud_rag = cloud_pipeline.ask(
                q, top_k=5, strategy="structural", model="gpt-4o"
            )
            cloud_time = time.time() - t0

            answer_cloud = cloud_rag["answer"][:400]
            print(f"    [{cloud_time:.1f}s, {cloud_rag['tokens']['output']} tok, "
                  f"chunks={cloud_rag['chunks_used']}]")
            print(wrap_text(answer_cloud, indent=8))

            row["cloud_rag"] = {
                "answer": cloud_rag["answer"],
                "time": round(cloud_time, 2),
                "tokens": cloud_rag["tokens"],
                "chunks_used": cloud_rag["chunks_used"],
                "citations": cloud_rag.get("citations", []),
                "sources": cloud_rag.get("sources", []),
                "dont_know": cloud_rag.get("dont_know", False),
            }
        else:
            print(f"\n  ▸ Облачная RAG: пропущена (нет OPENAI_API_KEY)")
            row["cloud_rag"] = None

        # ── Качество ──
        local_q = check_answer_quality(local_rag["answer"], item["expected"])
        row["local_quality"] = round(local_q, 2)
        print(f"\n  Качество (local): {local_q:.0%}")
        if row["cloud_rag"]:
            cloud_q = check_answer_quality(row["cloud_rag"]["answer"], item["expected"])
            row["cloud_quality"] = round(cloud_q, 2)
            print(f"  Качество (cloud): {cloud_q:.0%}")

        results.append(row)

    return results


# ── Этап 3: Тест "не знаю" (локальная) ───────────────────

def run_local_dont_know(local_pipeline):
    header("ЭТАП 3: ТЕСТ 'НЕ ЗНАЮ' (локальная RAG)")

    dont_know_results = []
    for i, q in enumerate(DONT_KNOW_QUESTIONS, 1):
        section(f"Нерелевантный {i}: {q}")

        t0 = time.time()
        rag = local_pipeline.ask_local(
            q, top_k=5, strategy="structural",
            ollama_model=OLLAMA_MODEL, min_similarity=0.3,
        )
        elapsed = time.time() - t0

        dont_know = rag.get("dont_know", False)
        max_sim = rag.get("max_similarity", None)
        print(f"  dont_know={dont_know} | max_sim={max_sim} | {elapsed:.1f}s")
        print(wrap_text(rag["answer"][:200]))

        status = "✓ ПРАВИЛЬНО" if dont_know else "✗ ОТВЕТИЛ (ожидали отказ)"
        print(f"  {status}")

        dont_know_results.append({
            "question": q,
            "dont_know": dont_know,
            "max_similarity": max_sim,
            "answer": rag["answer"],
            "correct": dont_know,
        })

    correct = sum(1 for r in dont_know_results if r["correct"])
    print(f"\n  Результат: {correct}/{len(dont_know_results)} правильных отказов")
    return dont_know_results


# ── Этап 4: Тест стабильности ─────────────────────────────

def run_stability_test(local_pipeline):
    header("ЭТАП 4: ТЕСТ СТАБИЛЬНОСТИ (3 запуска одного вопроса)")

    question = "Что такое машинное обучение?"
    answers = []
    times = []

    for run in range(1, 4):
        t0 = time.time()
        rag = local_pipeline.ask_local(
            question, top_k=5, strategy="structural", ollama_model=OLLAMA_MODEL,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        answers.append(rag["answer"])
        print(f"  Запуск {run}: {elapsed:.1f}s, {len(rag['answer'])} символов")

    # Сравниваем ответы
    from difflib import SequenceMatcher
    sims = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            ratio = SequenceMatcher(None, answers[i], answers[j]).ratio()
            sims.append(ratio)

    avg_sim = sum(sims) / len(sims) if sims else 0
    avg_time = sum(times) / len(times)
    time_std = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    print(f"\n  Средняя схожесть ответов: {avg_sim:.0%}")
    print(f"  Среднее время: {avg_time:.1f}s (±{time_std:.1f}s)")
    print(f"  Стабильность: {'✓ высокая' if avg_sim > 0.6 else '⚠ средняя' if avg_sim > 0.3 else '✗ низкая'}")

    return {
        "question": question,
        "runs": len(answers),
        "avg_similarity": round(avg_sim, 3),
        "avg_time": round(avg_time, 2),
        "time_std": round(time_std, 2),
        "times": [round(t, 2) for t in times],
    }


# ── Этап 5: Сводная таблица ──────────────────────────────

def print_summary(results, dont_know_results, stability):
    header("ЭТАП 5: СВОДНАЯ ТАБЛИЦА")

    has_cloud = any(r.get("cloud_rag") for r in results)

    # Таблица
    if has_cloud:
        print(f"\n  {'#':<3} {'Вопрос':<32} {'Local':>6} {'Cloud':>6} {'LQ':>5} {'CQ':>5} {'LT':>5} {'CT':>5}")
        print(f"  {'—'*3} {'—'*32} {'—'*6} {'—'*6} {'—'*5} {'—'*5} {'—'*5} {'—'*5}")
    else:
        print(f"\n  {'#':<3} {'Вопрос':<38} {'Tok':>6} {'Qual':>6} {'Time':>6}")
        print(f"  {'—'*3} {'—'*38} {'—'*6} {'—'*6} {'—'*6}")

    total_local_time = 0
    total_cloud_time = 0
    total_local_quality = 0
    total_cloud_quality = 0

    for i, r in enumerate(results, 1):
        q_short = r["question"][:30] + ".." if len(r["question"]) > 32 else r["question"]
        lt = r["local_rag"]["time"]
        lq = r.get("local_quality", 0)
        ltok = r["local_rag"]["tokens"]["output"]
        total_local_time += lt
        total_local_quality += lq

        if has_cloud and r.get("cloud_rag"):
            ct = r["cloud_rag"]["time"]
            cq = r.get("cloud_quality", 0)
            ctok = r["cloud_rag"]["tokens"]["output"]
            total_cloud_time += ct
            total_cloud_quality += cq
            print(f"  {i:<3} {q_short:<32} {ltok:>6} {ctok:>6} {lq:>5.0%} {cq:>5.0%} {lt:>4.1f}s {ct:>4.1f}s")
        else:
            print(f"  {i:<3} {q_short:<38} {ltok:>6} {lq:>5.0%} {lt:>5.1f}s")

    n = len(results)

    # Итоги
    section("Сравнение")
    print(f"  Локальная RAG ({OLLAMA_MODEL}):")
    print(f"    Среднее качество: {total_local_quality/n:.0%}")
    print(f"    Общее время: {total_local_time:.1f}s (среднее {total_local_time/n:.1f}s/вопрос)")

    if has_cloud:
        print(f"\n  Облачная RAG (gpt-4o):")
        print(f"    Среднее качество: {total_cloud_quality/n:.0%}")
        print(f"    Общее время: {total_cloud_time:.1f}s (среднее {total_cloud_time/n:.1f}s/вопрос)")

        speedup = total_cloud_time / total_local_time if total_local_time > 0 else 0
        if speedup > 1:
            print(f"\n  Локальная быстрее облачной в {speedup:.1f}x")
        else:
            print(f"\n  Облачная быстрее локальной в {1/speedup:.1f}x" if speedup > 0 else "")

    section("Стабильность")
    print(f"  Схожесть ответов (3 запуска): {stability['avg_similarity']:.0%}")
    print(f"  Время: {stability['avg_time']:.1f}s ± {stability['time_std']:.1f}s")

    section("Тест 'не знаю'")
    dk_correct = sum(1 for r in dont_know_results if r["correct"])
    print(f"  Правильных отказов: {dk_correct}/{len(dont_know_results)}")

    section("Выводы")
    print("  ▸ RAG-система полностью работает локально (Ollama)")
    print(f"  ▸ Модель: {OLLAMA_MODEL} (генерация) + {OLLAMA_EMBED_MODEL} (эмбеддинги)")
    print("  ▸ Никаких внешних API вызовов не требуется")
    if has_cloud:
        if total_local_quality / n >= total_cloud_quality / n * 0.7:
            print("  ▸ Качество локальной модели сопоставимо с облачной")
        else:
            print("  ▸ Облачная модель даёт более качественные ответы")


# ── Сохранение ────────────────────────────────────────────

def save_results(results, dont_know_results, stability, path):
    output = {
        "config": {
            "local_llm": OLLAMA_MODEL,
            "local_embedder": OLLAMA_EMBED_MODEL,
            "cloud_llm": "gpt-4o",
            "cloud_embedder": "text-embedding-3-small",
        },
        "benchmark": results,
        "dont_know": dont_know_results,
        "stability": stability,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  ✓ Результаты: {path}")


# ── Main ──────────────────────────────────────────────────

def main():
    print(f"\n{'#' * W}")
    print(f"##  ЛОКАЛЬНАЯ RAG (Ollama) vs ОБЛАЧНАЯ RAG (OpenAI)")
    print(f"##  Эмбеддинги + генерация — всё локально")
    print(f"{'#' * W}")

    # 1. Локальный индекс с локальными эмбеддингами
    local_db = os.path.join(PROJECT_ROOT, "rag_local_index.db")
    local_pipeline = build_local_index(local_db)
    if local_pipeline is None:
        print("  ✗ Не удалось построить локальный индекс")
        return False

    # 2. Облачный индекс (если есть ключ)
    cloud_pipeline = None
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if has_openai:
        cloud_db = os.path.join(PROJECT_ROOT, "rag_cloud_index.db")
        from indexing.embedder import Embedder
        cloud_pipeline = IndexingPipeline(db_path=cloud_db, embedder=Embedder())

        if not cloud_pipeline.store.is_indexed(DOC_FILE, strategy="structural"):
            print(f"\n  Индексация облачными эмбеддингами (OpenAI)...")
            with open(DOC_FILE, "r", encoding="utf-8") as f:
                text = f.read()
            cloud_pipeline.index_files(
                [(DOC_FILE, text)], strategy="structural", embed=True,
            )
        print(f"  ✓ Облачный индекс готов: {cloud_db}")
    else:
        print(f"\n  ⚠ OPENAI_API_KEY не найден — сравнение только локальное")

    # 3. Сравнение
    results = run_comparison(local_pipeline, cloud_pipeline)

    # 4. Тест "не знаю"
    dont_know_results = run_local_dont_know(local_pipeline)

    # 5. Стабильность
    stability = run_stability_test(local_pipeline)

    # 6. Итоги
    print_summary(results, dont_know_results, stability)

    # Сохраняем
    results_path = os.path.join(PROJECT_ROOT, "pipeline_output", "local_rag_benchmark.json")
    save_results(results, dont_know_results, stability, results_path)

    local_pipeline.close()
    if cloud_pipeline:
        cloud_pipeline.close()

    header("ГОТОВО")
    print("  ✓ Локальная RAG-система работает полностью автономно")
    print(f"  ✓ LLM: {OLLAMA_MODEL} | Embeddings: {OLLAMA_EMBED_MODEL}")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
