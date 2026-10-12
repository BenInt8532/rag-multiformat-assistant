"""
Чат с ассистентом в терминале.

Запуск: python app.py
Команды в чате: stats - сведения о базе и кешах, clear - очистить кеш ответов, exit - выход.
"""

import sys
from typing import Any, Dict

from assistant import Assistant
from config import get_settings, load_rules

LINE = "-" * 70
EXIT_COMMANDS = {"exit", "quit", "q", "выход"}
PREVIEW_LENGTH = 150


def describe_settings(settings: Dict[str, Any]) -> str:
    """Настройки двумя строками: ответ и подготовка документов."""
    budget = settings["context_budget"] if settings["context_budget"] is not None else "нет"
    threshold = settings["score_threshold"] if settings["score_threshold"] is not None else "нет"
    return (
        f"Ответ: модель {settings['model']}, температура {settings['temperature']}, "
        f"чанков не больше {settings['top_k']} (top_k), бюджет контекста {budget}, порог схожести {threshold}\n"
        f"Документы: нарезка {settings['chunking_mode']}, "
        f"очистка {'да' if settings['clean_text'] else 'нет'}, эмбеддинги {settings['embedding_model']}"
    )


def preview(text: str) -> str:
    """Начало чанка одной строкой; переносы строк внутри чанка показаны знаком " / "."""
    line = " / ".join(part.strip() for part in text.split("\n") if part.strip())
    return line[:PREVIEW_LENGTH] + "..." if len(line) > PREVIEW_LENGTH else line


def show_answer(result: Dict[str, Any]):
    """Вывод ответа, найденных чанков, времени и расхода токенов."""
    print(LINE)

    if result["from_cache"]:
        print(f"Источник ответа: кеш (сохранён {result['saved_at']})")
    elif result["model_called"]:
        print(f"Источник ответа: модель, чанков в контексте {len(result['context'])} из {len(result['found'])} найденных")
    else:
        print("Источник ответа: модель не вызывалась, подходящих чанков нет")

    print(f"\n{result['answer']}\n")
    if result["cited"]:
        print(f"Ответ взят из: {', '.join(result['cited'])}\n")

    if result["from_cache"]:
        files = sorted({chunk["source"] for chunk in result["context"]})
        print(f"Файлы в контексте: {', '.join(files) or 'нет'}")
    else:
        used = {chunk["id"] for chunk in result["context"]}
        print("Найденные чанки:")
        for number, chunk in enumerate(result["found"], 1):
            mark = "в контексте" if chunk["id"] in used else "отсеян"
            print(f"  {number}. схожесть {chunk['similarity']:.3f} | {chunk['source']} | {chunk['strategy']} | {mark}")
            print(f"     {preview(chunk['text'])}")

    timing = result["timing"]
    if result["from_cache"]:
        print(f"\nВремя: {timing['total']:.4f} с")
    else:
        print(f"\nВремя: {timing['total']:.2f} с (поиск {timing['search']:.2f} с, модель {timing['generation']:.2f} с)")
    if result["usage"]:
        print(f"Токены: запрос {result['usage']['prompt_tokens']}, ответ {result['usage']['completion_tokens']}")

    print(LINE)


def show_stats(assistant: Assistant):
    """Вывод сведений о базе, кешах и настройках."""
    stats = assistant.stats()
    index, answers = stats["index"], stats["answers"]

    print(LINE)
    print(f"База: коллекция {index['name']}, чанков {index['chunks']}")
    for source, count in sorted(index["sources"].items()):
        print(f"  {source}: {count}")

    print(f"\nКеш ответов: записей {answers['entries']}, файл {answers['size_mb']:.2f} МБ")
    if answers["entries"]:
        print(f"  первая запись {answers['first_saved']}, последняя {answers['last_saved']}")
    print(f"Кеш эмбеддингов: векторов {stats['embeddings']}")

    print(f"\n{describe_settings(stats['settings'])}")
    print(LINE)


def main():
    settings = get_settings()
    rules = load_rules()

    print("Ассистент по документам")
    print(describe_settings(settings))
    print()

    try:
        assistant = Assistant(settings, rules)
    except Exception as error:
        sys.exit(f"Не удалось запустить ассистента: {error}")

    print("\nГотов. Задайте вопрос. Команды: stats, clear, exit.\n")

    while True:
        try:
            question = input("Вопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        command = question.lower()
        if command in EXIT_COMMANDS:
            break
        if not question:
            continue
        if command == "stats":
            show_stats(assistant)
            continue
        if command == "clear":
            assistant.answers.clear()
            print("Кеш ответов очищен")
            continue

        try:
            show_answer(assistant.ask(question))
        except Exception as error:
            print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
