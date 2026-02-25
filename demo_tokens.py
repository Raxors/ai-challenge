"""
Демонстрация:
1. Как растут токены и стоимость по мере диалога
2. Что происходит при реальном переполнении контекста
"""

import textwrap
from dotenv import load_dotenv
from openai_model import OpenAIModel
from agent import Agent
from llm_interface import LLMError

load_dotenv()


def print_row(num, inp, out, hist, limit, pct, cost_req, cost_total):
    bar_len = 20
    filled = min(int(bar_len * pct / 100), bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(
        f"  {num:<4} {inp:>7} {out:>7} {hist:>8} / {limit:<8} "
        f"[{bar}] {pct:>5.1f}%  ${cost_req:>10.6f}  ${cost_total:>10.6f}"
    )


def print_wrapped(text, width=90, indent="    "):
    for paragraph in text.split("\n"):
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph, width=width, initial_indent=indent, subsequent_indent=indent)
            print(wrapped)
        else:
            print()


def demo_token_growth():
    """Часть 1: показываем рост токенов в диалоге."""
    print("=" * 100)
    print("ЧАСТЬ 1: РОСТ ТОКЕНОВ И СТОИМОСТИ ПО МЕРЕ ДИАЛОГА")
    print("=" * 100)
    print()

    llm = OpenAIModel(model="gpt-4o")
    agent = Agent(
        model=llm,
        model_name="gpt-4o",
        max_tokens=256,
        system_prompt="You are a helpful assistant. Keep answers short, under 50 words.",
        db_path="demo_history.db",
    )
    agent.reset()

    questions = [
        "What is Python?",
        "What are its main features?",
        "How does it compare to Java?",
        "What about performance differences?",
        "Can you summarize everything we discussed?",
        "Now explain it in more detail with examples.",
        "What are the best Python frameworks for web?",
        "Compare Django and Flask in detail.",
    ]

    header = f"  {'#':<4} {'Input':>7} {'Output':>7} {'History':>8}   {'Limit':<8} {'Context usage':<28} {'Req cost':>12}  {'Total cost':>12}"
    print(header)
    print("-" * 100)

    for q in questions:
        result = agent.ask(q)
        m = result["metrics"]
        print_row(
            m["request_number"],
            m["request_input_tokens"],
            m["request_output_tokens"],
            m["history_tokens"],
            m["context_limit"],
            m["usage_percent"],
            m["request_cost"],
            m["session_total_cost"],
        )

    print("-" * 100)
    print()
    print("  Обратите внимание:")
    print("  - Input tokens растёт с КАЖДЫМ запросом (вся история отправляется заново)")
    print("  - Стоимость каждого запроса УВЕЛИЧИВАЕТСЯ, даже если вопрос короткий")
    print(f"  - За {agent.request_number} запросов потрачено ${agent.session_cost:.6f}")
    print()

    agent.close()

    import os
    os.remove("demo_history.db")


def demo_overflow():
    """Часть 2: реальное переполнение контекста модели gpt-3.5-turbo (16 385 токенов)."""
    print("=" * 100)
    print("ЧАСТЬ 2: РЕАЛЬНОЕ ПЕРЕПОЛНЕНИЕ КОНТЕКСТА (gpt-3.5-turbo, лимит 16 385)")
    print("=" * 100)
    print()
    print("  Модель: gpt-3.5-turbo (контекст 16 385 токенов)")
    print("  max_tokens: 4096 (максимум на ответ)")
    print("  Доступно для истории: ~12 289 токенов")
    print("  Русский текст = больше токенов (кириллица ~2-3 токена на слово)")
    print()

    llm = OpenAIModel(model="gpt-3.5-turbo")
    agent = Agent(
        model=llm,
        model_name="gpt-3.5-turbo",
        max_tokens=4096,
        system_prompt=(
            "Ты — энциклопедический помощник. Отвечай максимально подробно и развёрнуто на русском языке. "
            "Никогда не сокращай ответ. Приводи примеры, даты, имена, формулы. "
            "Каждый ответ должен быть как статья в энциклопедии — не менее 1000 слов."
        ),
        db_path="demo_overflow.db",
    )
    agent.reset()

    long_messages = [
        "Опиши полную историю вычислительной техники от Чарльза Бэббиджа до современного искусственного интеллекта. Укажи каждый ключевой этап, прорыв, дату и имя изобретателя. Расскажи про механические калькуляторы, электронные лампы, транзисторы, интегральные схемы, микропроцессоры, персональные компьютеры, интернет и ИИ.",
        "Объясни квантовые вычисления максимально подробно. Расскажи про кубиты, суперпозицию, квантовую запутанность, квантовые вентили, коррекцию ошибок, квантовое превосходство, алгоритм Шора, алгоритм Гровера, текущее состояние квантовых компьютеров IBM, Google и других компаний.",
        "Сравни все основные языки программирования, созданные с 1950 года: Fortran, COBOL, Lisp, C, C++, Java, Python, JavaScript, Rust, Go, Kotlin, Swift. Для каждого языка укажи год создания, автора, парадигму, синтаксические особенности, области применения, преимущества и недостатки.",
        "Расскажи полную историю интернета от ARPANET до Web3. Опиши каждый протокол: TCP/IP, HTTP, HTTPS, DNS, FTP, SMTP, WebSocket. Объясни как работает маршрутизация, система доменных имён, SSL/TLS сертификаты, CDN, облачные вычисления и децентрализованные сети.",
        "Опиши все основные алгоритмы сортировки и поиска в информатике: пузырьковая сортировка, сортировка вставками, выбором, слиянием, быстрая сортировка, пирамидальная сортировка, поразрядная сортировка, сортировка подсчётом, линейный поиск, бинарный поиск, поиск в ширину, поиск в глубину. Для каждого приведи псевдокод и анализ сложности.",
        "Расскажи историю искусственного интеллекта от Алана Тьюринга до GPT-4. Опиши экспертные системы, машинное обучение, нейронные сети, глубокое обучение, свёрточные сети, рекуррентные сети, механизм внимания, трансформеры, BERT, GPT, диффузионные модели, reinforcement learning.",
        "Напиши полное руководство по базам данных: реляционные (MySQL, PostgreSQL, Oracle), NoSQL (MongoDB, Cassandra, Redis), графовые (Neo4j), векторные (Pinecone, Weaviate). Для каждой приведи примеры запросов, архитектуру, индексацию, репликацию, шардирование, транзакции ACID.",
        "Объясни всю криптографию от Цезаря до постквантовой. Опиши симметричное шифрование (AES, DES), асимметричное (RSA, ECC), хэш-функции (SHA-256, MD5), цифровые подписи, сертификаты X.509, протокол Диффи-Хеллмана, блокчейн, zero-knowledge proofs.",
    ]

    print("-" * 100)

    for i, msg in enumerate(long_messages, 1):
        try:
            print(f"\n  [Вопрос {i}]:")
            print_wrapped(msg)
            print()

            result = agent.ask(msg)
            m = result["metrics"]

            print(f"  [Ответ {i}]:")
            print_wrapped(result["text"])
            print()

            bar_len = 20
            filled = min(int(bar_len * m["usage_percent"] / 100), bar_len)
            bar = "#" * filled + "-" * (bar_len - filled)

            if m["usage_percent"] > 90:
                status = "!! CRITICAL"
            elif m["usage_percent"] > 70:
                status = "! WARNING"
            else:
                status = "OK"

            print(f"  --- Метрики запроса #{m['request_number']} ---")
            print(f"  Input: {m['request_input_tokens']}  Output: {m['request_output_tokens']}  "
                  f"История: {m['history_tokens']} / {m['context_limit']}  "
                  f"[{bar}] {m['usage_percent']:.1f}% {status}")
            print("-" * 100)

        except LLMError as e:
            print()
            print(f"  !!! ПЕРЕПОЛНЕНИЕ НА ЗАПРОСЕ #{i} !!!")
            print(f"  API ошибка: {e}")
            print("-" * 100)
            break

    print()
    print("  Что произошло при переполнении:")
    print("  - API вернул ошибку 400 (context_length_exceeded)")
    print("  - Запрос НЕ выполнен, но деньги за предыдущие запросы уже потрачены")
    print("  - Модель не может продолжить диалог без сброса истории")
    print("  - Русский текст занимает больше токенов (кириллица кодируется менее эффективно)")
    print()
    print(f"  Потрачено до ошибки: ${agent.session_cost:.6f}")
    print()

    agent.close()

    import os
    os.remove("demo_overflow.db")


if __name__ == "__main__":
    demo_token_growth()
    print()
    demo_overflow()
