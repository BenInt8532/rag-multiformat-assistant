"""Состав 48 прогонов с судьёй (12.10 04:50 - 14.10 21:31), на которых судья сравнивался с ключами. Без API.

Рабочие настройки: нарезка по типам с рабочими правилами (хэш правил как у прогона 2026-10-14-205258)
или единая 500/100, top_k 8, бюджет 2000. Нарочно испорченные: единая fixed; таблицы как текст
(stavki-po-geo.csv и offery.xlsx с tables=text); почти все файлы как текст (прогоны 205904, 210447, 211051).
Запуск из папки проекта: python statya-ocenka/sostav_progonov.py
"""
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "progony"
BAD_RULE_RUNS = ("2026-10-14-205904", "2026-10-14-210447", "2026-10-14-211051")


def h(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:5]


base_rules = h(json.loads((RESULTS / "2026-10-14-205258-rules-real-topk8-ragas.json").read_text(encoding="utf-8"))["rules"]["files"])
kinds = Counter()
for path in sorted(glob.glob(str(RESULTS / "*.json"))):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not any(isinstance(q.get("scores"), dict) and q["scores"].get("faithfulness") is not None for q in d.get("questions", [])):
        continue
    if not ("2026-10-12T04:50" <= d["created_at"] <= "2026-10-14T21:31"):
        continue
    s, r, name = d["settings"], d["rules"], Path(path).name
    mode = s["chunking_mode"]
    standard = s["top_k"] == 8 and s["context_budget"] == 2000
    if mode == "uniform":
        u = r["uniform"]
        if u.get("strategy") == "fixed":
            kind = "испорчена: fixed"
        elif standard and u == {"strategy": "paragraphs", "size": 500, "overlap": 100}:
            kind = "рабочая"
        else:
            kind = "опыт"
    else:
        files = r["files"]
        if name.startswith(BAD_RULE_RUNS):
            kind = "испорчена: почти всё как текст"
        elif files.get("offery.xlsx", {}).get("tables") == "text":
            kind = "испорчена: таблицы как текст"
        elif standard and h(files) == base_rules:
            kind = "рабочая"
        else:
            kind = "опыт"
    kinds[kind] += 1
    print(name[:17], mode, s["top_k"], s["context_budget"], kind)
print(sum(kinds.values()), dict(kinds))
