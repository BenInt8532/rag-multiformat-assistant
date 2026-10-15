"""
Разбор прогонов evaluate_ragas.py по ключам из klyuchi.py (без обращения к OpenAI).

Для каждого вопроса: верен ли ответ, был ли нужный кусок в контексте модели, был ли отказ.
Запуск из папки проекта:
    python statya-ocenka/razbor.py statya-ocenka/progony/файл1.json statya-ocenka/progony/файл2.json ...
    python statya-ocenka/razbor.py --detail statya-ocenka/progony/файл.json      (по вопросам, для сверки глазами)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from klyuchi import KEYS

sys.stdout.reconfigure(encoding="utf-8")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е"))


def has(text: str, option: str) -> bool:
    return any(norm(variant) in text for variant in option.split("|"))


def check(question: dict) -> dict:
    """Оценка одного ответа по ключам."""
    key = KEYS[question["user_input"]]
    refusal = bool(question["is_refusal"])
    if not key:  # вопрос вне базы: верен только отказ
        return {"correct": refusal, "found": None, "refusal": refusal}
    chunks = [norm(chunk) for chunk in question["contexts"]]
    found = all(any(all(norm(p) in chunk for p in group) for chunk in chunks) for group in key["ctx"])
    answer = norm(question["response"])
    correct = not refusal and all(has(answer, option) for option in key["ans"])
    return {"correct": correct, "found": found, "refusal": refusal}


def checkable(questions: list, name: str) -> list:
    """Вопросы, которые можно проверить: есть ключ и есть поле отказа (в ранних логах 12.10 его нет)."""
    ok = [q for q in questions if q["user_input"] in KEYS and "is_refusal" in q]
    if len(ok) < len(questions):
        print(f"[{name}] пропущено вопросов: {len(questions) - len(ok)} (нет ключа или поля is_refusal)", file=sys.stderr)
    return ok


def summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [check(q) for q in checkable(data["questions"], path.name)]
    base = [r for r in rows if r["found"] is not None]
    return {
        "file": path.name,
        "questions": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "found": sum(r["found"] for r in base),
        "base": len(base),
        "found_correct": sum(r["found"] and r["correct"] for r in base),
        "found_wrong": sum(r["found"] and not r["correct"] for r in base),
        "missed_correct": sum(not r["found"] and r["correct"] for r in base),
        "missed_wrong": sum(not r["found"] and not r["correct"] for r in base),
        "false_refusals": sum(r["refusal"] for r in base),
        "tokens": sum((q.get("usage") or {}).get("prompt_tokens", 0) for q in data["questions"]),
        # Запросы метаданных (М1, М4): токены запроса, отдельно от ответа
        "meta_tokens": sum((u or {}).get("prompt_tokens", 0)
                           for q in data["questions"] for u in ((q.get("meta") or {}).get("usage") or [])),
    }


def main():
    args = sys.argv[1:]
    detail = "--detail" in args
    paths = [Path(a) for a in args if a != "--detail"]
    if detail:
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"### {path.name}")
            for n, q in enumerate(data["questions"], 1):
                if q not in checkable([q], path.name):
                    continue
                r = check(q)
                mark = "+" if r["correct"] else "-"
                found = {True: "кусок есть", False: "куска нет", None: "вне базы"}[r["found"]]
                print(f"{n:>2} {mark} [{found}{', отказ' if r['refusal'] else ''}] {q['user_input']}")
                if not r["correct"]:
                    print(f"     Ответ: {q['response']}")
                    print(f"     Эталон: {q['reference']}")
            print()
        return
    print("| файл | верно | кусок найден | найден, верно | найден, неверно | не найден, верно | не найден, неверно "
          "| ложных отказов | токены | токены метаданных |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        s = summary(path)
        print(f"| {s['file']} | {s['correct']}/{s['questions']} | {s['found']}/{s['base']} | {s['found_correct']} "
              f"| {s['found_wrong']} | {s['missed_correct']} | {s['missed_wrong']} | {s['false_refusals']} "
              f"| {s['tokens']} | {s['meta_tokens']} |")


if __name__ == "__main__":
    main()
