import os
import logging
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Per-user conversation history (in-memory)
conversations: dict[int, list[dict]] = {}
MAX_HISTORY = 20


async def ollama_chat(user_id: int, user_message: str) -> str:
    """Send a message to the local Ollama model and return the response."""
    history = conversations.setdefault(user_id, [])
    history.append({"role": "user", "content": user_message})

    # Trim history to last MAX_HISTORY messages
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": history,
        "stream": False,
    }

    timeout = httpx.Timeout(10.0, read=600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

    assistant_text = data.get("message", {}).get("content", "")
    if not assistant_text:
        logger.warning("Empty response from Ollama: %s", data)
        assistant_text = "Модель не вернула ответ. Попробуйте ещё раз."
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversations.pop(user_id, None)
    await update.message.reply_text(
        f"Привет! Я бот, работающий на локальной LLM ({OLLAMA_MODEL}) через Ollama.\n"
        "Просто напиши мне сообщение, и я отвечу.\n"
        "/clear — очистить историю диалога"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversations.pop(user_id, None)
    await update.message.reply_text("История диалога очищена.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text

    # Show "typing" while waiting for Ollama
    await update.message.chat.send_action("typing")

    try:
        reply = await ollama_chat(user_id, user_text)
    except httpx.ConnectError:
        await update.message.reply_text(
            "Не удалось подключиться к Ollama. Убедитесь, что Ollama запущена "
            f"по адресу {OLLAMA_URL}."
        )
        return
    except Exception as e:
        logger.error("Ollama error: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при обращении к модели: {e}")
        return

    # Telegram has a 4096 char limit per message
    if len(reply) <= 4096:
        await update.message.reply_text(reply)
    else:
        for i in range(0, len(reply), 4096):
            await update.message.reply_text(reply[i : i + 4096])


def main() -> None:
    if not TELEGRAM_TOKEN:
        print("Ошибка: установите TELEGRAM_TOKEN в файле .env")
        print("Получить токен можно у @BotFather в Telegram.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started — model: %s, ollama: %s", OLLAMA_MODEL, OLLAMA_URL)
    app.run_polling()


if __name__ == "__main__":
    main()
