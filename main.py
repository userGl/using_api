from dotenv import load_dotenv
from pathlib import Path
import os
import time
from openai import OpenAI


base_dir = Path(__file__).resolve().parent
load_dotenv(base_dir / ".env")


def log_print(text):
    print(text)
    with (base_dir / "chat.log").open("a", encoding="utf-8") as log:
        log.write(text + "\n")


folder_id = os.environ["YANDEX_FOLDER_ID"]
model_name = os.environ["YANDEX_MODEL"]


temperature = float(os.getenv("YANDEX_TEMPERATURE", "0.3"))
max_output_tokens = int(os.getenv("YANDEX_MAX_OUTPUT_TOKENS", "1000"))
top_p = float(os.getenv("YANDEX_TOP_P", "1.0"))


prompt_path = base_dir / "prompt.txt"
instructions = prompt_path.read_text(encoding="utf-8").strip()


client = OpenAI(
    api_key=os.environ["YANDEX_API_KEY"],
    base_url="https://ai.api.cloud.yandex.net/v1",
    project=folder_id,
)


log_print(f"\n--- Запуск: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
log_print(f"Выбранная модель: {model_name}")
log_print(f"Промпт: {instructions}")


while True:
    question = input("\nВведите вопрос (0 — выход): ").strip()

    if question == "0":
        log_print("Выход из программы.")
        break

    started = time.perf_counter()

    response = client.responses.create(
        model=f"gpt://{folder_id}/{model_name}",
        instructions=instructions,
        input=question,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
    )

    elapsed = time.perf_counter() - started

    log_print(
        f"\nВопрос: {question}"
        f"\nОтвет: {response.output_text}"
        f"\nВремя полного ответа: {elapsed:.2f} с"
        f"\nТемпература в ответе API: {response.temperature}"
        f"\nTop_p в ответе API: {response.top_p}"
        f"\nЛимит выходных токенов: {response.max_output_tokens}"
    )

    if response.usage is not None:
        log_print("--- Расход токенов ---")
        log_print(f"Входные токены:  {response.usage.input_tokens}")
        log_print(f"Выходные токены: {response.usage.output_tokens}")
        log_print(f"Всего токенов:  {response.usage.total_tokens}")
    else:
        log_print("\nAPI не вернул статистику токенов.")

    if response.status == "incomplete":
        log_print("\nОтвет не завершён.")
        log_print(f"Причина: {response.incomplete_details}")