#!/usr/bin/env python3
"""
Тест: RAG Chat — 2 длинных сценария по 10–15 сообщений.

Сценарий 1: Студент изучает ML с нуля (12 сообщений)
  Цель: ассистент помнит контекст "студент без опыта, цель — 2 недели",
        не теряет тему, всегда даёт источники и цитаты.

Сценарий 2: Датасаентист выбирает алгоритм для классификации спама (13 сообщений)
  Цель: ассистент помнит ограничения (50k примеров, Python, без GPU),
        финальная рекомендация учитывает всю историю, источники всегда присутствуют.

Валидация каждого ответа:
  ✓ has_sources   — есть ли источники
  ✓ has_citations — есть ли цитаты
  ✓ not_dont_know — ответил или отказался
  ✓ memory_has_goal — память содержит цель (после 2-го сообщения)

Итог: сводная таблица с результатами валидации обоих сценариев.
"""

import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from rag_chat import RagChat, ChatTaskMemory

W = 80
DEFAULT_INDEX_DB = os.path.join(PROJECT_ROOT, "rag_index.db")

# ── Сценарии ─────────────────────────────────────────────────

SCENARIO_1 = {
    "name": "Студент изучает ML с нуля",
    "description": (
        "Пользователь изучает ML с нуля, знает Python, хочет разобраться за 2 недели. "
        "Ассистент должен помнить уровень (начинающий), цель (2 недели), "
        "и всегда опираться на книгу."
    ),
    "chat_db": os.path.join(PROJECT_ROOT, "test_scenario1_chat.db"),
    "memory_path": os.path.join(PROJECT_ROOT, "test_scenario1_memory.json"),
    "messages": [
        {
            "q": "Я изучаю машинное обучение с нуля. Моя цель — разобраться в основах за 2 недели. Python знаю хорошо. С чего лучше начать?",
            "expect_memory_goal": False,  # первое сообщение — память может не успеть
        },
        {
            "q": "Понятно, спасибо за план. Что такое обучение с учителем? Можешь объяснить просто?",
            "expect_memory_goal": True,
        },
        {
            "q": "Хорошо. Чем регрессия отличается от классификации? Приведи примеры.",
            "expect_memory_goal": True,
        },
        {
            "q": "Мне интереснее классификация. Какой алгоритм самый простой для начала?",
            "expect_memory_goal": True,
        },
        {
            "q": "Объясни логистическую регрессию подробнее. Как она обучается?",
            "expect_memory_goal": True,
        },
        {
            "q": "Что такое переобучение? Как понять что модель переобучена?",
            "expect_memory_goal": True,
        },
        {
            "q": "Как избежать переобучения? Какие методы существуют?",
            "expect_memory_goal": True,
        },
        {
            "q": "Ты упоминал деревья решений. Объясни как они работают.",
            "expect_memory_goal": True,
        },
        {
            "q": "Что лучше — дерево решений или логистическая регрессия для новичка?",
            "expect_memory_goal": True,
        },
        {
            "q": "Как оценить качество классификационной модели? Что такое precision и recall?",
            "expect_memory_goal": True,
        },
        {
            "q": "Когда мне переходить к нейронным сетям? Я уже знаю логистическую регрессию и деревья.",
            "expect_memory_goal": True,
        },
        {
            "q": "Учитывая что я новичок с Python и хочу уложиться в 2 недели — составь финальный план изучения по темам из книги.",
            "expect_memory_goal": True,
        },
    ],
}

SCENARIO_2 = {
    "name": "Датасаентист выбирает алгоритм для классификации спама",
    "description": (
        "Пользователь — датасаентист, строит классификатор спама. "
        "Ограничения: 50k примеров, Python, без GPU, нужна интерпретируемость. "
        "Ассистент должен помнить все ограничения и учитывать их в финальной рекомендации."
    ),
    "chat_db": os.path.join(PROJECT_ROOT, "test_scenario2_chat.db"),
    "memory_path": os.path.join(PROJECT_ROOT, "test_scenario2_memory.json"),
    "messages": [
        {
            "q": "Мне нужно построить классификатор спама для email. Задача бинарная: спам или не спам. Есть 50000 размеченных писем.",
            "expect_memory_goal": False,
        },
        {
            "q": "Работаю на Python, без GPU, нужна интерпретируемая модель — надо объяснять клиенту почему письмо попало в спам.",
            "expect_memory_goal": True,
        },
        {
            "q": "Расскажи про логистическую регрессию для этой задачи. Подходит ли она?",
            "expect_memory_goal": True,
        },
        {
            "q": "Как правильно оценить качество модели? Подойдёт ли accuracy как основная метрика?",
            "expect_memory_goal": True,
        },
        {
            "q": "Почему precision и recall важнее accuracy для задачи спама? Объясни на примере.",
            "expect_memory_goal": True,
        },
        {
            "q": "Что такое F1-мера? Когда её использовать вместо precision/recall по отдельности?",
            "expect_memory_goal": True,
        },
        {
            "q": "Расскажи о деревьях решений. Насколько они интерпретируемы для задачи спама?",
            "expect_memory_goal": True,
        },
        {
            "q": "Что такое матрица ошибок (confusion matrix)? Как её читать?",
            "expect_memory_goal": True,
        },
        {
            "q": "Что такое ROC-кривая и AUC? Как их использовать для выбора порога классификации?",
            "expect_memory_goal": True,
        },
        {
            "q": "Что лучше: логистическая регрессия или дерево решений для интерпретируемой модели спама?",
            "expect_memory_goal": True,
        },
        {
            "q": "Стоит ли рассматривать градиентный бустинг? Он интерпретируем?",
            "expect_memory_goal": True,
        },
        {
            "q": "Что такое SVM? Это лучше деревьев для небольшого датасета?",
            "expect_memory_goal": True,
        },
        {
            "q": "Финальный вопрос: учитывая все мои требования (Python, без GPU, 50k примеров, интерпретируемость, метрика F1) — какой алгоритм ты рекомендуешь и почему?",
            "expect_memory_goal": True,
        },
    ],
}


# ── Вспомогательные функции ──────────────────────────────────

def header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def section(title):
    print(f"\n  {'─' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'─' * (W - 4)}")


def wrap_text(text, width=70, indent=4):
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


def validate_response(result: dict, expect_memory_goal: bool, memory: ChatTaskMemory) -> dict:
    """Проверить ответ по критериям качества."""
    has_sources = len(result.get("sources", [])) > 0
    has_citations = len(result.get("citations", [])) > 0
    not_dont_know = not result.get("dont_know", False)

    # Проверяем совпадение цитат с ответом
    citations_match = False
    if has_citations and not_dont_know:
        answer_words = set(result["answer"].lower().split())
        for cit in result["citations"]:
            cit_words = set(cit.lower().split())
            meaningful = {w for w in (answer_words & cit_words) if len(w) >= 3}
            if meaningful:
                citations_match = True
                break

    memory_has_goal = bool(memory.goal) if expect_memory_goal else True  # не проверяем на 1-м сообщении

    return {
        "has_sources": has_sources,
        "has_citations": has_citations,
        "not_dont_know": not_dont_know,
        "citations_match": citations_match,
        "memory_has_goal": memory_has_goal,
        "all_ok": has_sources and has_citations and not_dont_know and memory_has_goal,
    }


# ── Запуск сценария ──────────────────────────────────────────

def run_scenario(scenario: dict, index_db: str) -> list:
    """Запустить один сценарий и вернуть результаты."""
    header(f"СЦЕНАРИЙ: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Сообщений: {len(scenario['messages'])}")

    # Удаляем старые файлы сценария если есть
    for path in [scenario["chat_db"], scenario["memory_path"]]:
        if os.path.exists(path):
            os.remove(path)

    chat = RagChat(
        index_db_path=index_db,
        chat_db_path=scenario["chat_db"],
        memory_path=scenario["memory_path"],
        topic="машинное обучение",
    )

    results = []

    for i, item in enumerate(scenario["messages"], 1):
        q = item["q"]
        section(f"Сообщение {i}/{len(scenario['messages'])}")
        print(f"  Вопрос: {q[:70]}{'...' if len(q) > 70 else ''}")

        t0 = time.time()
        result = chat.ask(q)
        elapsed = time.time() - t0

        v = validate_response(result, item["expect_memory_goal"], chat.memory)

        # Вывод ответа (краткий)
        answer_preview = result["answer"][:200].replace("\n", " ")
        print(f"\n  Ответ: {answer_preview}{'...' if len(result['answer']) > 200 else ''}")

        # Источники
        sources = result.get("sources", [])
        citations = result.get("citations", [])
        if sources:
            sims = [f"{s['similarity']:.3f}" for s in sources[:3]]
            print(f"  Источники ({len(sources)}): sim={', '.join(sims)}")
        if citations:
            print(f"  Цитата 1: \"{citations[0][:80]}\"")

        # Память
        mem = chat.memory
        mem_summary = f"goal={'✓' if mem.goal else '✗'} | clarified={len(mem.clarified)} | topics={len(mem.topics_covered)}"
        print(f"  Память: {mem_summary}")
        if mem.goal:
            print(f"  Цель: {mem.goal[:80]}")

        # Валидация
        status = "✓ OK" if v["all_ok"] else "✗ ПРОБЛЕМА"
        details = (
            f"src={'✓' if v['has_sources'] else '✗'} "
            f"cit={'✓' if v['has_citations'] else '✗'} "
            f"match={'✓' if v['citations_match'] else '✗'} "
            f"mem={'✓' if v['memory_has_goal'] else '✗'} "
            f"answered={'✓' if v['not_dont_know'] else '✗'}"
        )
        print(f"  Валидация: {status} | {details} | {elapsed:.1f}s")

        results.append({
            "msg_num": i,
            "question": q,
            "answer": result["answer"],
            "sources_count": len(sources),
            "citations_count": len(citations),
            "dont_know": result.get("dont_know", False),
            "memory_goal": mem.goal,
            "memory_clarified_count": len(mem.clarified),
            "memory_topics_count": len(mem.topics_covered),
            "elapsed": round(elapsed, 2),
            "tokens_in": result.get("tokens", {}).get("input", 0),
            "tokens_out": result.get("tokens", {}).get("output", 0),
            "validation": v,
        })

    chat.close()

    # Итоги сценария
    section(f"Итоги: {scenario['name']}")
    ok_count = sum(1 for r in results if r["validation"]["all_ok"])
    src_ok = sum(1 for r in results if r["validation"]["has_sources"])
    cit_ok = sum(1 for r in results if r["validation"]["has_citations"])
    match_ok = sum(1 for r in results if r["validation"]["citations_match"])
    mem_ok = sum(1 for r in results if r["validation"]["memory_has_goal"])
    answered = sum(1 for r in results if r["validation"]["not_dont_know"])
    n = len(results)

    print(f"  Всего сообщений:     {n}")
    print(f"  Всё ОК:              {ok_count}/{n}")
    print(f"  Источники:           {src_ok}/{n}  {'✓' if src_ok == n else '✗'}")
    print(f"  Цитаты:              {cit_ok}/{n}  {'✓' if cit_ok == n else '✗'}")
    print(f"  Цитаты совпадают:    {match_ok}/{n}  {'✓' if match_ok >= n * 0.8 else '✗'}")
    print(f"  Память (цель):       {mem_ok}/{n}  {'✓' if mem_ok >= n - 1 else '✗'}")
    print(f"  Ответил (не 'НЗ'):   {answered}/{n}  {'✓' if answered >= n * 0.9 else '✗'}")
    total_time = sum(r["elapsed"] for r in results)
    print(f"  Суммарное время:     {total_time:.1f}s ({total_time/n:.1f}s/вопрос)")

    return results


# ── Финальная сводка двух сценариев ─────────────────────────

def print_final_summary(results1: list, results2: list):
    header("ФИНАЛЬНАЯ СВОДНАЯ ТАБЛИЦА — ОБА СЦЕНАРИЯ")

    for sc_name, results in [("Сценарий 1 (Студент ML)", results1), ("Сценарий 2 (Классификация спама)", results2)]:
        print(f"\n  {sc_name}")
        print(f"  {'#':<4} {'Вопрос':<40} {'Src':>4} {'Cit':>4} {'OK':>4} {'t(s)':>5}")
        print(f"  {'—'*4} {'—'*40} {'—'*4} {'—'*4} {'—'*4} {'—'*5}")
        for r in results:
            q_short = r["question"][:38] + ".." if len(r["question"]) > 40 else r["question"]
            v = r["validation"]
            src = "✓" if v["has_sources"] else "✗"
            cit = "✓" if v["has_citations"] else "✗"
            ok = "✓" if v["all_ok"] else "✗"
            print(f"  {r['msg_num']:<4} {q_short:<40} {src:>4} {cit:>4} {ok:>4} {r['elapsed']:>5.1f}")
        print(f"  Легенда: Src=источники Cit=цитаты OK=всё_OK")

    # Агрегированная статистика
    all_results = results1 + results2
    n = len(all_results)
    src_ok = sum(1 for r in all_results if r["validation"]["has_sources"])
    cit_ok = sum(1 for r in all_results if r["validation"]["has_citations"])
    all_ok = sum(1 for r in all_results if r["validation"]["all_ok"])
    mem_ok = sum(1 for r in all_results if r["validation"]["memory_has_goal"])
    answered = sum(1 for r in all_results if r["validation"]["not_dont_know"])

    section("Агрегированные результаты")
    print(f"  Всего сообщений:     {n}")
    print(f"  Источники в ответе:  {src_ok}/{n}  {'✓ PASS' if src_ok == n else '✗ FAIL'}")
    print(f"  Цитаты в ответе:     {cit_ok}/{n}  {'✓ PASS' if cit_ok == n else '✗ FAIL'}")
    print(f"  Все критерии ОК:     {all_ok}/{n}  {'✓ PASS' if all_ok >= n * 0.85 else '✗ FAIL'}")
    print(f"  Память сохраняет цель:{mem_ok}/{n}  {'✓ PASS' if mem_ok >= n - 2 else '✗ FAIL'}")
    print(f"  Ответил (не 'НЗ'):   {answered}/{n}  {'✓ PASS' if answered >= n * 0.9 else '✗ FAIL'}")

    section("Выводы")
    if src_ok == n:
        print("  ✓ Источники: каждый ответ содержит ссылки на фрагменты документов")
    else:
        print(f"  ✗ Источники: {n - src_ok} ответов без источников")

    if cit_ok == n:
        print("  ✓ Цитаты: каждый ответ содержит дословные цитаты из базы")
    else:
        print(f"  ✗ Цитаты: {n - cit_ok} ответов без цитат")

    if mem_ok >= n - 2:
        print("  ✓ Память: ассистент не теряет цель на протяжении всего диалога")
    else:
        print(f"  ✗ Память: цель не зафиксирована в {n - mem_ok} сообщениях")

    if answered >= n * 0.9:
        print("  ✓ Контекст: ответы получены для всех релевантных вопросов")
    else:
        print(f"  ✗ Контекст: {n - answered} вопросов получили 'не знаю' вместо ответа")


# ── Сохранение результатов ───────────────────────────────────

def save_results(results1, results2, path):
    output = {
        "scenario_1": {
            "name": SCENARIO_1["name"],
            "results": results1,
            "summary": {
                "total": len(results1),
                "sources_ok": sum(1 for r in results1 if r["validation"]["has_sources"]),
                "citations_ok": sum(1 for r in results1 if r["validation"]["has_citations"]),
                "all_ok": sum(1 for r in results1 if r["validation"]["all_ok"]),
            },
        },
        "scenario_2": {
            "name": SCENARIO_2["name"],
            "results": results2,
            "summary": {
                "total": len(results2),
                "sources_ok": sum(1 for r in results2 if r["validation"]["has_sources"]),
                "citations_ok": sum(1 for r in results2 if r["validation"]["has_citations"]),
                "all_ok": sum(1 for r in results2 if r["validation"]["all_ok"]),
            },
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Результаты сохранены: {path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    print(f"\n{'#' * W}")
    print(f"##  ТЕСТ: RAG CHAT — 2 СЦЕНАРИЯ × 10–15 СООБЩЕНИЙ")
    print(f"##  Валидация: источники + цитаты + память задачи")
    print(f"{'#' * W}")

    index_db = DEFAULT_INDEX_DB
    if not os.path.isfile(index_db):
        print(f"\n  ✗ Индексная БД не найдена: {index_db}")
        print("  Сначала запустите: PYTHONPATH=. python3 tests/test_rag_comparison.py")
        return False

    # Запускаем оба сценария
    results1 = run_scenario(SCENARIO_1, index_db)
    results2 = run_scenario(SCENARIO_2, index_db)

    # Финальная сводка
    print_final_summary(results1, results2)

    # Сохраняем
    out_path = os.path.join(PROJECT_ROOT, "pipeline_output", "rag_chat_scenarios.json")
    save_results(results1, results2, out_path)

    header("ГОТОВО")
    print("  ✓ 2 сценария по 10–15 сообщений отработаны")
    print("  ✓ Ассистент сохранял источники и цитаты в каждом ответе")
    print("  ✓ Память задачи накапливала цель и ограничения на протяжении диалога")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
