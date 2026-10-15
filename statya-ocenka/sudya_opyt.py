"""
Опыт для статьи 2: что делает с оценкой Faithfulness подпись источника и отказ (ночь 2026-10-14).

Берётся готовый прогон (ответы и куски уже есть, модель ответа не вызывается). Для каждого вопроса по базе,
на который ассистент ответил (не отказ), судье RAGAS отдаются три варианта при тех же кусках:
  A - ответ как есть (без строки источников, как в рабочей оценке);
  B - тот же ответ с подписью в конце «(файл: имя)», как писала модель по первой версии промпта;
  C - вместо ответа фраза отказа «В документах нет информации по этому вопросу.»
Каждый вариант оценивается REPEATS раз. Результат - statya-ocenka/opyty/sudya-opyt.json и сводка на экран.

Запуск из папки проекта:  python statya-ocenka/sudya_opyt.py statya-ocenka/progony/2026-10-14-215644-rules-real-topk8-ragas.json
"""
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from config import load_env
from prompts import NO_CONTEXT_ANSWER

REPEATS = 3
JUDGE_MODEL = "gpt-4o-mini"


async def main(path: str):
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Не найден OPENAI_API_KEY (должен быть в .env проекта)")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [q for q in data["questions"] if q.get("expected_source") and not q["is_refusal"] and q["contexts"]]
    results = []
    quota_hit = False
    # Результат статьи (opyty/sudya-opyt.json) повтор не затирает: пишет рядом файл со временем в имени
    out = Path(__file__).parent / "opyty" / "sudya-opyt.json"
    if out.exists():
        out = out.with_name(f"sudya-opyt-{time.strftime('%Y%m%d-%H%M%S')}.json")
    try:
        async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=5, timeout=120) as client:
            metric = Faithfulness(llm=llm_factory(JUDGE_MODEL, client=client, max_tokens=4096))
            for n, q in enumerate(items, 1):
                files = ", ".join(q["cited_sources"]) or q["expected_source"]
                variants = {
                    "A": q["response"],
                    "B": f"{q['response']} (файл: {files})",
                    "C": NO_CONTEXT_ANSWER,
                }
                row = {"question": q["user_input"], "scores": {}}
                for name, text in variants.items():
                    scores = []
                    for _ in range(REPEATS):
                        try:
                            res = await metric.ascore(user_input=q["user_input"], response=text,
                                                      retrieved_contexts=q["contexts"])
                            scores.append(None if res.value is None else round(float(res.value), 4))
                        except Exception as error:  # noqa: BLE001 - сбой одной оценки не должен терять остальные
                            # Печатается только тип ошибки: в тексте ошибки OpenAI бывает часть ключа
                            print(f"  [!] {n} {name}: {type(error).__name__}")
                            if "insufficient_quota" in str(error):
                                quota_hit = True
                                break
                            scores.append(None)
                    row["scores"][name] = scores
                    if quota_hit:
                        break
                if quota_hit:
                    print("Квота OpenAI кончилась: опыт остановлен, посчитанное сохранено.")
                    break
                results.append(row)
                print(f"{n}/{len(items)} A={row['scores']['A']} B={row['scores']['B']} C={row['scores']['C']}")
    finally:
        out.write_text(json.dumps({"source_run": path, "repeats": REPEATS, "judge": JUDGE_MODEL,
                                   "interrupted_by_quota": quota_hit, "rows": results},
                                  ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nСводка (среднее по всем оценкам; доля вопросов, где 3 повтора дали разные оценки):")
    for name in "ABC":
        values = [v for r in results for v in r["scores"][name] if v is not None]
        if not values:
            print(f"  {name}: оценок нет")
            continue
        unstable = sum(1 for r in results if len(set(r["scores"][name])) > 1)
        dist = {k: sum(1 for v in values if v == k) for k in (0.0, 1.0)}
        print(f"  {name}: среднее {statistics.mean(values):.3f}, оценок {len(values)}, из них 0.0: {dist[0.0]}, 1.0: {dist[1.0]}; "
              f"нестабильных вопросов {unstable} из {len(results)}")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
