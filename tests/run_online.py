"""
Проверки с ключом OpenAI: сборка базы, контрольные вопросы, кеш эмбеддингов,
отключение файла, порог схожести. Тратит деньги на счёте OpenAI (по оценке, центы).

Запуск из папки проекта:
    python tests/run_online.py

Результат печатается и сохраняется в results/<дата>-online-checks.json.
"""

import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from assistant import Assistant  # noqa: E402
from config import RESULTS_DIR, get_settings, load_rules  # noqa: E402

# Контрольные вопросы: по одному на файл базы.
# expected - слова, хотя бы одно из которых должно быть в верном ответе
CONTROL_QUESTIONS = [
    ("glossariy.txt", "Что такое апрув?", ["подтвер"]),
    ("pravila-raboty.md", "Можно ли привлекать мотивированный трафик?", ["запрещ", "нельзя", "не разреш"]),
    ("faq.html", "Сколько раз отправляется постбэк?", ["триж", "три раза", "3 раза"]),
    ("vyplaty-i-hold.pdf", "Какой срок холда на финансовых офферах у вебмастера со статусом «Проверенный»?", ["7"]),
    ("offery.xlsx", "Какая ставка у оффера Карта Выгода?", ["2100", "2 100"]),
    ("stavki-po-geo.csv", "Какая ставка за лид по офферу Школа кода в Узбекистане?", ["900"]),
    ("pervaya-kampaniya.docx", "Какой шаг в инструкции по запуску первой кампании идёт следующим после шага с настройкой постбэка?", ["креатив"]),
    ("tracker_ru.py", "При каком ROI скрипт трекера считает кампанию убыточной?", ["20"]),
    ("tracker_en.py", "Сколько визитов должна набрать группа, чтобы скрипт проверил, ставить ли её на паузу?", ["150"]),
]

OUT_OF_BASE_QUESTION = "Какая столица Франции?"
SWITCHED_OFF_FILE = "glossariy.txt"


def build(settings, rules):
    """Запуск ассистента без служебной печати; возвращает ассистента и сведения о сборке базы."""
    started = time.perf_counter()
    assistant = Assistant(settings, rules, log=lambda message: None)
    info = {key: value for key, value in assistant.build_info.items() if key != "fingerprint"}
    info["seconds"] = round(time.perf_counter() - started, 2)
    info["read_seconds"] = round(info["read_seconds"], 2)
    return assistant, info


def describe_build(info) -> str:
    """Сведения о сборке одной строкой."""
    if not info["rebuilt"]:
        return f"время {info['seconds']} с, чанков {info['chunks']}, база не пересобиралась"
    return (f"время {info['seconds']} с, чанков {info['chunks']}, база пересобрана: "
            f"из кеша {info['from_cache']}, посчитано заново {info['computed']}")


def main():
    settings = get_settings()
    rules = load_rules()
    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "settings": settings, "checks": {}}

    print("=" * 70)
    print("ПРОВЕРКИ С КЛЮЧОМ")
    print("=" * 70)
    print(f"Python {sys.version.split()[0]}")
    print(f"Настройки: {settings}\n")

    # ---------- Проверка 7: первый запуск ----------
    assistant, first = build(settings, rules)
    sources = assistant.stats()["index"]["sources"]
    print("[7] Сборка базы")
    print(f"    {describe_build(first)}")
    for source, count in sorted(sources.items()):
        print(f"    - {source}: {count}")
    report["checks"]["7_build"] = {**first, "sources": sources}

    # ---------- Проверка 8: контрольные вопросы ----------
    print("\n[8] Контрольные вопросы: по одному на файл")
    rows = []
    for source, question, expected in CONTROL_QUESTIONS:
        if source not in sources:
            print(f"    пропущен (файла нет в базе): {source}")
            continue

        result = assistant.ask(question, use_cache=False)
        found = [chunk["source"] for chunk in result["context"]]
        answer_ok = any(word.lower() in result["answer"].lower() for word in expected)
        row = {
            "source": source,
            "question": question,
            "answer": result["answer"],
            "expected_words": expected,
            "answer_has_expected": answer_ok,
            "hit": source in found,
            "hit_first": bool(found) and found[0] == source,
            "found_sources": found,
            "similarities": [round(chunk["similarity"], 3) for chunk in result["context"]],
            "seconds": round(result["timing"]["total"], 2),
            "usage": result["usage"]
        }
        rows.append(row)
        print(f"\n    Файл: {source}")
        print(f"    Вопрос: {question}")
        print(f"    Ответ: {row['answer']}")
        print(f"    Чанки: {found}  схожесть {row['similarities']}")
        print(f"    Нужный файл первым: {'да' if row['hit_first'] else 'НЕТ'} | "
              f"в ответе есть ожидаемое ({'/'.join(expected)}): {'да' if answer_ok else 'НЕТ'}")
    report["checks"]["8_control_questions"] = rows

    hits = sum(row["hit_first"] for row in rows)
    answers = sum(row["answer_has_expected"] for row in rows)
    print(f"\n    Итог: нужный файл первым {hits} из {len(rows)}, ожидаемое в ответе {answers} из {len(rows)}")

    # ---------- Проверка 9: повторный запуск ----------
    _, second = build(settings, rules)
    print("\n[9] Повторный запуск без изменений")
    print(f"    {describe_build(second)}")
    report["checks"]["9_second_start"] = second

    # ---------- Проверка 10: выключить один файл ----------
    reduced_rules = copy.deepcopy(rules)
    reduced_rules["files"].pop(SWITCHED_OFF_FILE, None)
    reduced, off = build(settings, reduced_rules)
    left = reduced.stats()["index"]["sources"].get(SWITCHED_OFF_FILE, 0)
    print(f"\n[10] Файл {SWITCHED_OFF_FILE} выключен")
    print(f"    его чанков в базе: {left}; {describe_build(off)}")
    report["checks"]["10_file_off"] = {**off, "chunks_of_switched_off": left}

    # Возврат полной базы: все векторы должны прийти из кеша
    _, restored = build(settings, rules)
    print(f"    файл включён обратно: {describe_build(restored)}")
    report["checks"]["10_file_on_again"] = restored

    # ---------- Проверка 11: вопрос не по базе, с порогом и без ----------
    # Порог в обеих половинах задаётся явно: результат не должен зависеть от SCORE_THRESHOLD в .env
    print("\n[11] Вопрос не по базе")
    no_threshold, _ = build({**settings, "score_threshold": None}, rules)
    plain = no_threshold.ask(OUT_OF_BASE_QUESTION, use_cache=False)
    similarities = [round(chunk["similarity"], 3) for chunk in plain["found"]]
    print(f"    без порога: модель вызывалась: {'да' if plain['model_called'] else 'нет'}, схожесть {similarities}")
    print(f"    ответ: {plain['answer']}")

    strict, _ = build({**settings, "score_threshold": 0.2}, rules)
    guarded = strict.ask(OUT_OF_BASE_QUESTION, use_cache=False)
    print(f"    с порогом 0.2: модель вызывалась: {'да' if guarded['model_called'] else 'нет'}")
    print(f"    ответ: {guarded['answer']}")
    report["checks"]["11_out_of_base"] = {
        "without_threshold": {"model_called": plain["model_called"], "answer": plain["answer"],
                              "similarities": similarities, "usage": plain["usage"]},
        "with_threshold_0.2": {"model_called": guarded["model_called"], "answer": guarded["answer"]}
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y-%m-%d-%H%M%S}-online-checks.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Готово. Результаты сохранены: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
