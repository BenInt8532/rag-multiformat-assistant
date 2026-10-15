"""
Общая часть трёх опытов с судьёй RAGAS (статья 2, rag-evaluation, 2026-10-15).

Не запускается сама. Её подключают opyt_otkazy.py, opyt_podpis.py и opyt_shum.py, лежащие рядом.
Что здесь: подготовка (папка проекта, ключ, импорты), судья Faithfulness с ограничением
числа параллельных запросов и остановкой при insufficient_quota, чтение прогона, запись JSON.

Папка проекта - текущая папка запуска (там должны лежать config.py и prompts.py).
Скрипты опытов можно не копировать в проект: запуск из папки проекта по полному пути к скрипту.
"""
import asyncio
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path.cwd()
JUDGE_MODEL = "gpt-4o-mini"
LOGS = Path(__file__).resolve().parent / "opyty"


def prepare():
    """Папка проекта в путях импорта, вывод в UTF-8, ключ из .env (файл здесь не читается и не выводится)."""
    if not (PROJECT / "config.py").exists() or not (PROJECT / "prompts.py").exists():
        sys.exit(f"Запускать надо из папки проекта (где config.py и prompts.py), сейчас: {PROJECT}")
    sys.path.insert(0, str(PROJECT))
    sys.stdout.reconfigure(encoding="utf-8")
    from config import load_env
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Не найден OPENAI_API_KEY (должен быть в .env проекта)")


def load_run(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_question(run: dict, text: str) -> dict:
    """Вопрос прогона по точному тексту. Если его нет - остановка с понятным сообщением."""
    for q in run["questions"]:
        if q["user_input"] == text:
            return q
    sys.exit(f"В прогоне нет вопроса: {text!r}. Нужен прогон из списка в шапке скрипта.")


class Judge:
    """Faithfulness из RAGAS: одна оценка = 2 запроса к gpt-4o-mini (разбор ответа на утверждения и проверка их по кускам)."""

    def __init__(self, parallel: int):
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=5, timeout=120)
        self._metric = Faithfulness(llm=llm_factory(JUDGE_MODEL, client=self._client, max_tokens=4096))
        self._sem = asyncio.Semaphore(parallel)
        self.calls = 0          # сколько раз вызван ascore
        self.errors = 0         # сколько оценок не получено (кроме квоты)
        self.quota_hit = False  # квота кончилась: новые запросы не отправляются

    async def close(self):
        await self._client.close()

    async def score(self, question: str, response: str, contexts: list, tag: str = ""):
        """Одна оценка 0..1 или None (сбой, пустой разбор или остановка из-за квоты)."""
        if self.quota_hit:
            return None
        async with self._sem:
            if self.quota_hit:
                return None
            self.calls += 1
            try:
                res = await self._metric.ascore(user_input=question, response=response,
                                                retrieved_contexts=contexts)
            except Exception as error:  # noqa: BLE001 - любая ошибка API не должна терять посчитанное
                text = f"{error!r} {error}"
                if "insufficient_quota" in text:
                    self.quota_hit = True
                    print(f"\n[!] {tag}: insufficient_quota - квота OpenAI кончилась. Новые запросы остановлены, "
                          f"то, что успело посчитаться, будет сохранено.")
                else:
                    self.errors += 1
                    print(f"  [!] {tag}: {type(error).__name__}")  # текст ошибки OpenAI может содержать часть ключа
                return None
            value = res.value
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return round(float(value), 4)


def save(name: str, payload: dict) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / name
    if out.exists():  # результат статьи повтор не затирает: пишет рядом файл со временем в имени
        out = out.with_name(f"{out.stem}-{time.strftime('%Y%m%d-%H%M%S')}{out.suffix}")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def describe(values: list) -> str:
    """Короткая сводка списка оценок: среднее, число 0.0 и 1.0, размах."""
    values = [v for v in values if v is not None]
    if not values:
        return "нет оценок"
    return (f"среднее {statistics.mean(values):.3f}, оценок {len(values)}, "
            f"0.0: {sum(1 for v in values if v == 0.0)}, 1.0: {sum(1 for v in values if v == 1.0)}")


def spread(values: list) -> float:
    """Размах (максимум минус минимум) по непустым оценкам."""
    values = [v for v in values if v is not None]
    return max(values) - min(values) if values else 0.0


def distribution(values: list) -> str:
    cnt = Counter(v for v in values if v is not None)
    return ", ".join(f"{k}:{n}" for k, n in sorted(cnt.items()))


def parse_args(default_repeats: int):
    """Аргументы: путь к прогону, --repeats N, --parallel N."""
    args = sys.argv[1:]
    repeats, parallel, paths = default_repeats, 4, []
    i = 0
    while i < len(args):
        if args[i] == "--repeats":
            repeats = int(args[i + 1]); i += 2
        elif args[i] == "--parallel":
            parallel = int(args[i + 1]); i += 2
        else:
            paths.append(args[i]); i += 1
    if len(paths) != 1:
        sys.exit("Нужен один аргумент - путь к прогону statya-ocenka/progony/<файл>.json (см. шапку скрипта)")
    return paths[0], repeats, parallel
