"""
Судья RAGAS против ключей (статья «Оценка RAG: где ошибается RAGAS и как его проверить»). Без API.

Берёт прогоны с оценками судьи и по каждому ответу сравнивает: верен ли ответ по ключам (razbor.check)
и какие Faithfulness и Context Precision поставил судья. Запуск из папки проекта:
    python statya-ocenka/sudya_vs_klyuchi.py statya-ocenka/progony/файл1.json ...
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from razbor import KEYS, check

sys.stdout.reconfigure(encoding="utf-8")

rows = []
runs = 0
for arg in sys.argv[1:]:
    data = json.loads(Path(arg).read_text(encoding="utf-8"))
    # Прогоны без судьи (RAGAS_SKIP=1) не берутся
    if not any(v is not None for q in data["questions"] for v in (q.get("scores") or {}).values()):
        continue
    runs += 1
    for q in data["questions"]:
        if q["user_input"] not in KEYS or not q.get("scores"):
            continue
        r = check(q)
        if r["found"] is None:
            continue  # вне базы
        rows.append({**r, "f": q["scores"].get("faithfulness"), "cp": q["scores"].get("context_precision"),
                     "q": q["user_input"]})

def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return "нет оценок"
    hi = sum(v >= 0.9 for v in values)
    return f"n={len(values)}, среднее {statistics.mean(values):.2f}, >=0.9 у {hi} ({hi / len(values):.0%})"

print(f"Прогонов с судьёй: {runs}, ответов по базе: {len(rows)}, разных вопросов: {len({r['q'] for r in rows})}")
answered = [r for r in rows if not r["refusal"]]
print("Faithfulness, ответ верный (не отказ):", stat([r["f"] for r in answered if r["correct"]]))
print("Faithfulness, ответ НЕверный (не отказ):", stat([r["f"] for r in answered if not r["correct"]]))
print("Context Precision, кусок найден:", stat([r["cp"] for r in rows if r["found"]]))
print("Context Precision, кусок НЕ найден:", stat([r["cp"] for r in rows if not r["found"]]))
print("Context Precision, ложный отказ:", stat([r["cp"] for r in rows if r["refusal"]]))
# Неверные ответы с высоким Faithfulness: судья доволен, а ответ неверен
bad = [r for r in answered if not r["correct"] and r["f"] is not None and r["f"] >= 0.9]
print(f"\nНеверных ответов с Faithfulness >= 0.9: {len(bad)}")
for r in bad[:40]:
    print(f"  f={r['f']} cp={r['cp']} найден={r['found']} | {r['q']}")
