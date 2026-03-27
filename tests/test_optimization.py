#!/usr/bin/env python3
"""
Оптимизация локальной LLM для RAG.

Оси оптимизации:
  1. Параметры: temperature, num_predict, num_ctx, top_p, repeat_penalty
  2. Квантование: Q4_K_M (базовая) vs Q8_0 (качество) vs Q2_K (скорость)
  3. Промпт: JSON (базовый) vs Plain text (оптимизированный)

Метрики:
  - Качество ответов (keyword overlap)
  - Скорость (время генерации)
  - Потребление ресурсов (токены, размер модели)
"""

import os
import sys
import time
import json
import subprocess
import re
from difflib import SequenceMatcher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from indexing.pipeline import IndexingPipeline
from indexing.ollama_embedder import OllamaEmbedder

W = 80
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DOC_FILE = os.path.join(DOCS_DIR, "ml_textbook.md")

OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ── Бенчмарк вопросы ──────────────────────────────────────
BENCHMARK = [
    {
        "q": "Что такое машинное обучение?",
        "expected_keywords": ["область", "искусственного", "интеллекта", "алгоритмы",
                              "учатся", "данных", "закономерности", "паттерны"],
    },
    {
        "q": "Чем отличается контролируемое обучение от неконтролируемого?",
        "expected_keywords": ["размеченные", "метку", "предсказывать", "неразмеченные",
                              "кластеры", "структуру", "аномалии"],
    },
    {
        "q": "Что такое переобучение и как с ним бороться?",
        "expected_keywords": ["запоминает", "шум", "обобщает", "регуляризация",
                              "dropout", "ранняя", "остановка"],
    },
    {
        "q": "Как работает линейная регрессия?",
        "expected_keywords": ["прямую", "линию", "зависимость", "минимизируем",
                              "среднеквадратичную", "градиентного"],
    },
    {
        "q": "Что такое нейронные сети и как они обучаются?",
        "expected_keywords": ["слоёв", "нейронов", "веса", "активации",
                              "обратное", "распространение", "градиент"],
    },
]

DONT_KNOW_Q = "Каков рецепт итальянской пасты карбонара?"


# ── Конфигурации для сравнения ─────────────────────────────

CONFIGS = {
    "baseline": {
        "label": "Базовая (temperature=0.3, JSON промпт)",
        "method": "ask_local",
        "ollama_options": None,
        "description": "Текущая конфигурация без изменений",
    },
    "opt_temp01": {
        "label": "temperature=0.1",
        "method": "ask_local_optimized",
        "ollama_options": {"temperature": 0.1, "num_predict": 512},
        "description": "Низкая температура для детерминированности",
    },
    "opt_temp01_ctx4k": {
        "label": "temp=0.1, ctx=4096",
        "method": "ask_local_optimized",
        "ollama_options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
        "description": "Сокращённый контекст для скорости",
    },
    "opt_full": {
        "label": "Полная оптимизация",
        "method": "ask_local_optimized",
        "ollama_options": {
            "temperature": 0.1,
            "num_predict": 512,
            "num_ctx": 4096,
            "top_p": 0.8,
            "repeat_penalty": 1.2,
        },
        "description": "Все параметры оптимизированы + plain text промпт",
    },
}


def header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def section(title):
    print(f"\n  {'-' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'-' * (W - 4)}")


def wrap_text(text, width=68, indent=8):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(" " * indent + cur)
            cur = w
        else:
            cur = cur + " " + w if cur else w
    if cur:
        lines.append(" " * indent + cur)
    return "\n".join(lines)


def check_quality(answer, keywords):
    """Качество = доля найденных ключевых слов в ответе."""
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found / len(keywords) if keywords else 0


# ── Этап 1: Подготовка индекса ────────────────────────────

def prepare_index():
    header("ЭТАП 1: ПОДГОТОВКА ИНДЕКСА")

    db_path = os.path.join(PROJECT_ROOT, "rag_local_index.db")
    embedder = OllamaEmbedder(model=OLLAMA_EMBED_MODEL)
    pipeline = IndexingPipeline(db_path=db_path, embedder=embedder)

    if pipeline.store.is_indexed(DOC_FILE, strategy="structural"):
        print(f"  ✓ Индекс уже существует")
    else:
        print(f"  Индексация {os.path.basename(DOC_FILE)}...")
        with open(DOC_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        pipeline.index_files([(DOC_FILE, text)], strategy="structural", embed=True)
        print(f"  ✓ Проиндексировано")

    return pipeline


# ── Этап 2: Тест параметров ──────────────────────────────

def run_config(pipeline, config_name, config, model="qwen3.5:latest"):
    """Запускает один конфиг на всех вопросах, возвращает метрики."""
    method_name = config["method"]
    options = config.get("ollama_options")

    results = []
    total_time = 0
    total_quality = 0
    total_out_tokens = 0

    for item in BENCHMARK:
        q = item["q"]
        t0 = time.time()

        if method_name == "ask_local":
            rag = pipeline.ask_local(
                q, top_k=5, strategy="structural", ollama_model=model,
            )
        elif method_name == "ask_local_optimized":
            rag = pipeline.ask_local_optimized(
                q, top_k=5, strategy="structural", ollama_model=model,
                ollama_options=options,
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        elapsed = time.time() - t0
        quality = check_quality(rag["answer"], item["expected_keywords"])

        results.append({
            "question": q,
            "answer": rag["answer"],
            "time": round(elapsed, 2),
            "quality": round(quality, 3),
            "out_tokens": rag["tokens"]["output"],
            "in_tokens": rag["tokens"]["input"],
            "citations": rag.get("citations", []),
            "dont_know": rag.get("dont_know", False),
        })
        total_time += elapsed
        total_quality += quality
        total_out_tokens += rag["tokens"]["output"]

    # Тест "не знаю"
    t0 = time.time()
    if method_name == "ask_local":
        dk = pipeline.ask_local(
            DONT_KNOW_Q, top_k=5, strategy="structural", ollama_model=model,
        )
    else:
        dk = pipeline.ask_local_optimized(
            DONT_KNOW_Q, top_k=5, strategy="structural", ollama_model=model,
            ollama_options=options,
        )
    dk_time = time.time() - t0
    dk_correct = dk.get("dont_know", False)

    n = len(BENCHMARK)
    return {
        "config_name": config_name,
        "label": config["label"],
        "results": results,
        "avg_quality": round(total_quality / n, 3),
        "avg_time": round(total_time / n, 2),
        "total_time": round(total_time, 2),
        "avg_out_tokens": round(total_out_tokens / n),
        "dont_know_correct": dk_correct,
        "dont_know_time": round(dk_time, 2),
    }


def run_parameter_optimization(pipeline):
    header("ЭТАП 2: ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")

    all_results = {}

    for name, config in CONFIGS.items():
        section(f"{config['label']} ({name})")
        print(f"  {config['description']}")
        if config.get("ollama_options"):
            print(f"  Параметры: {config['ollama_options']}")
        print()

        metrics = run_config(pipeline, name, config)

        # Печатаем результаты по каждому вопросу
        for r in metrics["results"]:
            q_short = r["question"][:35] + ".." if len(r["question"]) > 37 else r["question"]
            print(f"    {q_short:<38} Q={r['quality']:.0%}  T={r['time']:.1f}s  "
                  f"Tok={r['out_tokens']}")

        dk_status = "✓" if metrics["dont_know_correct"] else "✗"
        print(f"\n  СРЕДНЕЕ: качество={metrics['avg_quality']:.0%}, "
              f"время={metrics['avg_time']:.1f}s/вопрос, "
              f"токены={metrics['avg_out_tokens']}")
        print(f"  Не знаю: {dk_status} ({metrics['dont_know_time']:.1f}s)")

        all_results[name] = metrics

    return all_results


# ── Этап 3: Квантование ──────────────────────────────────

def run_model_size_comparison(pipeline):
    """Сравнение моделей разного размера (аналог квантования)."""
    header("ЭТАП 3: СРАВНЕНИЕ МОДЕЛЕЙ РАЗНОГО РАЗМЕРА")

    models = [
        {"name": "qwen3:1.7b",    "tag": "qwen3:1.7b",    "size_gb": 1.4, "params": "1.7B"},
        {"name": "qwen3:4b",      "tag": "qwen3:4b",      "size_gb": 2.5, "params": "4B"},
        {"name": "qwen3.5:latest", "tag": "qwen3.5:latest", "size_gb": 6.6, "params": "9.7B"},
    ]

    opt_options = {
        "temperature": 0.1,
        "num_predict": 512,
        "num_ctx": 4096,
        "top_p": 0.8,
        "repeat_penalty": 1.2,
    }

    model_results = {}

    for model_info in models:
        tag = model_info["tag"]
        section(f"{model_info['params']} — {tag} ({model_info['size_gb']} GB)")

        # Проверяем наличие
        check = subprocess.run(["ollama", "show", tag, "--modelfile"],
                               capture_output=True, text=True)
        if check.returncode != 0:
            print(f"  ✗ Модель не установлена")
            continue

        total_time = 0
        total_quality = 0
        question_results = []

        for item in BENCHMARK:
            q = item["q"]
            t0 = time.time()
            rag = pipeline.ask_local_optimized(
                q, top_k=5, strategy="structural", ollama_model=tag,
                ollama_options=opt_options,
            )
            elapsed = time.time() - t0
            quality = check_quality(rag["answer"], item["expected_keywords"])
            total_time += elapsed
            total_quality += quality

            q_short = q[:35] + ".." if len(q) > 37 else q
            print(f"    {q_short:<38} Q={quality:.0%}  T={elapsed:.1f}s  "
                  f"Tok={rag['tokens']['output']}")
            question_results.append({
                "question": q,
                "quality": round(quality, 3),
                "time": round(elapsed, 2),
                "tokens": rag["tokens"]["output"],
                "answer": rag["answer"][:200],
            })

        # Тест "не знаю"
        t0 = time.time()
        dk = pipeline.ask_local_optimized(
            DONT_KNOW_Q, top_k=5, strategy="structural", ollama_model=tag,
            ollama_options=opt_options,
        )
        dk_time = time.time() - t0
        dk_correct = dk.get("dont_know", False)

        n = len(BENCHMARK)
        avg_q = total_quality / n
        avg_t = total_time / n

        dk_icon = "✓" if dk_correct else "✗"
        print(f"\n  СРЕДНЕЕ: качество={avg_q:.0%}, время={avg_t:.1f}s, "
              f"не знаю: {dk_icon} ({dk_time:.1f}s)")

        model_results[model_info["name"]] = {
            "tag": tag,
            "params": model_info["params"],
            "size_gb": model_info["size_gb"],
            "avg_quality": round(avg_q, 3),
            "avg_time": round(avg_t, 2),
            "total_time": round(total_time, 2),
            "dont_know_correct": dk_correct,
            "questions": question_results,
        }

    return model_results


# ── Этап 4: Тест стабильности оптимизированной модели ────

def run_stability_test(pipeline):
    header("ЭТАП 4: СТАБИЛЬНОСТЬ ОПТИМИЗИРОВАННОЙ МОДЕЛИ")

    question = "Что такое машинное обучение?"
    opt_options = {
        "temperature": 0.1,
        "num_predict": 512,
        "num_ctx": 4096,
        "top_p": 0.8,
        "repeat_penalty": 1.2,
    }

    answers = []
    times = []

    for run in range(1, 4):
        t0 = time.time()
        rag = pipeline.ask_local_optimized(
            question, top_k=5, strategy="structural",
            ollama_model="qwen3.5:latest",
            ollama_options=opt_options,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        answers.append(rag["answer"])
        print(f"  Запуск {run}: {elapsed:.1f}s, {len(rag['answer'])} символов, "
              f"{rag['tokens']['output']} токенов")

    sims = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            sims.append(SequenceMatcher(None, answers[i], answers[j]).ratio())

    avg_sim = sum(sims) / len(sims) if sims else 0
    avg_time = sum(times) / len(times)
    time_std = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    print(f"\n  Схожесть: {avg_sim:.0%}")
    print(f"  Время: {avg_time:.1f}s ± {time_std:.1f}s")
    print(f"  Стабильность: {'✓ высокая' if avg_sim > 0.7 else '⚠ средняя'}")

    return {
        "avg_similarity": round(avg_sim, 3),
        "avg_time": round(avg_time, 2),
        "time_std": round(time_std, 2),
    }


# ── Этап 5: Итоговое сравнение ───────────────────────────

def print_final_comparison(param_results, model_results, stability):
    header("ЭТАП 5: ИТОГОВОЕ СРАВНЕНИЕ")

    # Таблица параметров
    section("Оптимизация параметров и промпта")
    print(f"\n  {'Конфиг':<25} {'Качество':>9} {'Время':>8} {'Tok':>5} {'DK':>4}")
    print(f"  {'—'*25} {'—'*9} {'—'*8} {'—'*5} {'—'*4}")

    baseline_time = param_results.get("baseline", {}).get("avg_time", 1)
    baseline_quality = param_results.get("baseline", {}).get("avg_quality", 0)

    for name, m in param_results.items():
        dk = "✓" if m["dont_know_correct"] else "✗"
        marker = " ★" if name == "opt_full" else ""
        print(f"  {m['label'][:25]:<25} {m['avg_quality']:>8.0%} {m['avg_time']:>6.1f}s "
              f"{m['avg_out_tokens']:>5} {dk:>4}{marker}")

    # Улучшение
    opt = param_results.get("opt_full", {})
    if opt and baseline_quality > 0:
        quality_delta = (opt["avg_quality"] - baseline_quality) / baseline_quality * 100
        speed_delta = (baseline_time - opt["avg_time"]) / baseline_time * 100
        section("Улучшение: baseline → opt_full")
        print(f"  Качество: {baseline_quality:.0%} → {opt['avg_quality']:.0%} "
              f"({'+' if quality_delta >= 0 else ''}{quality_delta:.0f}%)")
        print(f"  Скорость: {baseline_time:.1f}s → {opt['avg_time']:.1f}s "
              f"({'+' if speed_delta >= 0 else ''}{speed_delta:.0f}%)")
        print(f"  Токены: {param_results['baseline']['avg_out_tokens']} → "
              f"{opt['avg_out_tokens']} выход/вопрос")

    # Таблица моделей разного размера
    if model_results:
        section("Сравнение моделей (размер vs качество vs скорость)")
        print(f"\n  {'Модель':<18} {'Размер':>8} {'Качество':>9} {'Время':>8} {'DK':>4}")
        print(f"  {'—'*18} {'—'*8} {'—'*9} {'—'*8} {'—'*4}")

        best_q = max(m["avg_quality"] for m in model_results.values())
        for name, mr in model_results.items():
            dk = "✓" if mr["dont_know_correct"] else "✗"
            marker = " ★" if mr["avg_quality"] == best_q else ""
            print(f"  {mr['params'] + ' ' + name:<18} {mr['size_gb']:>6.1f}GB "
                  f"{mr['avg_quality']:>8.0%} {mr['avg_time']:>6.1f}s {dk:>4}{marker}")

    # Стабильность
    section("Стабильность оптимизированной модели")
    print(f"  Схожесть (3 запуска): {stability['avg_similarity']:.0%}")
    print(f"  Время: {stability['avg_time']:.1f}s ± {stability['time_std']:.1f}s")

    # Рекомендации
    section("Рекомендации")
    # Найдём лучшее соотношение качество/скорость
    if model_results:
        best = max(model_results.items(),
                   key=lambda x: x[1]["avg_quality"] / max(x[1]["avg_time"], 0.1))
        print(f"  Лучшее соотношение качество/скорость: {best[0]} ({best[1]['params']})")
        print(f"    Качество: {best[1]['avg_quality']:.0%}, Скорость: {best[1]['avg_time']:.1f}s")
    print()
    print("  Оптимальная конфигурация для RAG:")
    print("    temperature: 0.1   — детерминированность")
    print("    num_predict: 512   — ограничение длины")
    print("    num_ctx:     4096  — ускорение KV cache")
    print("    top_p:       0.8   — фокусированный sampling")
    print("    repeat_penalty: 1.2 — снижение повторов")
    print("    Промпт:      Plain text с few-shot, 3-5 предложений")


# ── Сохранение ────────────────────────────────────────────

def save_results(param_results, model_results, stability, path):
    output = {
        "parameter_optimization": param_results,
        "model_size_comparison": model_results,
        "stability": stability,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  ✓ Результаты: {path}")


# ── Main ──────────────────────────────────────────────────

def main():
    print(f"\n{'#' * W}")
    print(f"##  ОПТИМИЗАЦИЯ ЛОКАЛЬНОЙ LLM ДЛЯ RAG")
    print(f"##  qwen3.5 (9.7B, Q4_K_M) + nomic-embed-text")
    print(f"{'#' * W}")

    # 1. Подготовка
    pipeline = prepare_index()

    # 2. Оптимизация параметров
    param_results = run_parameter_optimization(pipeline)

    # 3. Модели разного размера
    model_results = run_model_size_comparison(pipeline)

    # 4. Стабильность
    stability = run_stability_test(pipeline)

    # 5. Итог
    print_final_comparison(param_results, model_results, stability)

    # Сохраняем
    results_path = os.path.join(
        PROJECT_ROOT, "pipeline_output", "optimization_benchmark.json"
    )
    save_results(param_results, model_results, stability, results_path)

    pipeline.close()

    header("ГОТОВО")
    print("  ✓ Оптимизация завершена")
    print()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
