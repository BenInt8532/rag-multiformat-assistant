"""
Отчёт по ловушкам на уровне чанков, без ключа и без сети.

Для каждой ловушки проверяется, оказались ли нужные куски текста в одном чанке:
при единой нарезке (uniform) и при нарезке по правилам (rules). Это не тест "прошёл / не прошёл",
а наблюдение: ловушка могла и не сработать, и это тоже результат.

Запуск из папки проекта:
    python tests/report_traps.py
"""

import ast
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from tests.test_offline import DATA_DIR, chunks_of  # noqa: E402

# Ловушка: файл, что проверяем, и куски текста, которые должны быть в одном чанке
TRAPS = [
    ("vyplaty-i-hold.pdf", "ответ в двух абзацах: холд и его сокращение",
     ["составляет 14 дней", "Указанный срок сокращается вдвое"]),
    ("vyplaty-i-hold.pdf", "ответ в двух абзацах: минимальная сумма и крипта",
     ["Минимальная сумма вывода составляет 3000 рублей", "этот порог выше в пять раз"]),
    ("pravila-raboty.md", "смысл в заголовке: запрещённые источники",
     ["Запрещённые источники трафика", "Мотивированный трафик"]),
    ("pravila-raboty.md", "смысл в заголовке: последний пункт перечня",
     ["Запрещённые источники трафика", "Cashback-сервисы"]),
    ("faq.html", "вопрос и конец длинного ответа",
     ["Как настроить постбэк?", "Отправить тестовый постбэк"]),
    ("faq.html", "вопрос и конец ответа из трёх абзацев",
     ["Что делать, если статистика в трекере и в сети расходится?", "выгрузку кликов"]),
    ("stavki-po-geo.csv", "строка таблицы и название столбца",
     ["Узбекистан", "Ставка за лид, руб"]),
    ("offery.xlsx", "строка таблицы и название столбца",
     ["Курс аналитика", "Холд, дней"]),
    ("pervaya-kampaniya.docx", "список из десяти шагов целиком",
     ["1. Выберите один оффер", "10. Посчитайте ROI"]),
    ("pervaya-kampaniya.docx", "шаг с отсылкой и шаг, на который он ссылается",
     ["Настройте постбэк в разделе", "После этого подготовьте"]),
]


def together(chunks, pieces) -> bool:
    """Есть ли чанк, в котором встречаются все куски текста."""
    return any(all(piece in chunk for piece in pieces) for chunk in chunks)


def split_functions(file_name: str, uniform: bool):
    """Функции файла с кодом, у которых первая и последняя строки оказались в разных чанках."""
    source = (DATA_DIR / file_name).read_text(encoding="utf-8")
    lines = source.split("\n")
    chunks = chunks_of(file_name, uniform=uniform)

    broken = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            first = lines[node.lineno - 1].strip()
            last = lines[node.end_lineno - 1].strip()
            if not together(chunks, [first, last]):
                broken.append(node.name)
    return broken


def main():
    print(f"{'Файл':<24} {'Ловушка':<52} {'uniform':<10} {'rules':<10}")
    print("-" * 98)

    for file_name, title, pieces in TRAPS:
        results = []
        for uniform in (True, False):
            results.append("вместе" if together(chunks_of(file_name, uniform=uniform), pieces) else "РАЗОРВАНО")
        print(f"{file_name:<24} {title:<52} {results[0]:<10} {results[1]:<10}")

    print("\nФункции кода, разрезанные между чанками:")
    for file_name in ("tracker_ru.py", "tracker_en.py"):
        for uniform in (True, False):
            mode = "uniform" if uniform else "rules"
            broken = split_functions(file_name, uniform)
            print(f"  {file_name:<16} {mode:<8} {len(broken)}: {', '.join(broken) or 'нет'}")

    print("\nЧисло чанков по файлам:")
    files = sorted({file_name for file_name, _, _ in TRAPS} | {"glossariy.txt", "tracker_ru.py", "tracker_en.py"})
    for file_name in files:
        uniform_count = len(chunks_of(file_name, uniform=True))
        rules_count = len(chunks_of(file_name, uniform=False))
        print(f"  {file_name:<24} uniform {uniform_count:>3}   rules {rules_count:>3}")


if __name__ == "__main__":
    main()
