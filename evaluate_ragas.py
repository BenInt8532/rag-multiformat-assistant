"""
Оценка качества RAG системы через RAGAS.
Использует OpenAI API для RAG и для метрик RAGAS.

Вопросы, эталонные ответы и ожидаемый файл-источник лежат в eval/questions.json.
Кроме метрик RAGAS считается попадание: нашёл ли поиск чанк из нужного файла и на каком месте.
Результаты каждого прогона сохраняются в папку results.
"""

import asyncio
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    Faithfulness,
)

from config import QUESTIONS_PATH, RESULTS_DIR, get_settings, load_rules
from assistant import Assistant
from prompts import NO_CONTEXT_ANSWER

# Настройки из .env файла и правила нарезки из chunking.txt
settings = get_settings()
rules = load_rules()

# Модель-судья и эмбеддинги для метрик. Заданы отдельно от модели ассистента:
# при смене OPENAI_MODEL оценки остаются сравнимыми
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"
# Запас на ответ судьи: по умолчанию в RAGAS 1024 токена, длинный разбор в них не помещается
JUDGE_MAX_TOKENS = 4096
JUDGE_MAX_RETRIES = 5
JUDGE_TIMEOUT = 120  # секунд на один запрос

# Откуда брать эталон: real - из файла вопросов, answer - первые 100 символов ответа модели
GROUND_TRUTH_MODE = (os.getenv("GROUND_TRUTH_MODE") or "real").strip().lower()
if GROUND_TRUTH_MODE not in ("real", "answer"):
    sys.exit(f"Ошибка в .env: GROUND_TRUTH_MODE должно быть real или answer, а не '{GROUND_TRUTH_MODE}'")

# Метрики в порядке вывода: ключ -> подпись
METRIC_LABELS = {
    "faithfulness": "Faithfulness (точность ответа)",
    "context_precision": "Context Precision (точность контекста)",
    "answer_relevancy": "Answer Relevancy (ответ по вопросу)"
}


def load_questions() -> List[Dict[str, Any]]:
    """
    Чтение вопросов для оценки.

    Returns:
        список словарей: question - вопрос, reference - эталонный ответ,
        source - файл, из которого должен прийти ответ (null для вопроса вне базы),
        trap - какая ловушка проверяется (необязательно)
    """
    if not QUESTIONS_PATH.exists():
        sys.exit(f"Не найден файл вопросов: {QUESTIONS_PATH}")

    try:
        with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"Ошибка в {QUESTIONS_PATH.name}: {e}")

    for i, item in enumerate(questions, 1):
        for name in ("question", "reference"):
            if not item.get(name):
                sys.exit(f"В {QUESTIONS_PATH.name} у вопроса {i} нет поля '{name}'")
    return questions


def collect_samples(assistant: Assistant, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Получение ответов RAG системы и найденных чанков по каждому вопросу.

    Args:
        assistant: ассистент, который отвечает на вопросы
        questions: вопросы из файла вопросов

    Returns:
        список словарей: вопрос, ответ, чанки контекста, эталон, ожидаемый файл и попадание
    """
    samples = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, item in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {item['question']}")

        # Получаем ответ от RAG системы (без использования кеша)
        result = assistant.ask(item["question"], use_cache=False)

        # Ground truth - эталонный ответ: подготовленный вручную или часть ответа модели
        if GROUND_TRUTH_MODE == "answer":
            reference = result["answer"][:100]
        else:
            reference = item["reference"]

        # Попадание: место первого чанка из ожидаемого файла среди чанков контекста
        expected_source = item.get("source")
        context_sources = [chunk["source"] for chunk in result["context"]]
        hit_rank = None
        if expected_source in context_sources:
            hit_rank = context_sources.index(expected_source) + 1

        samples.append({
            "user_input": item["question"],
            "response": result["answer"],
            # Контекст - список текстов чанков, которые пошли в промпт
            "retrieved_contexts": [chunk["text"] for chunk in result["context"]],
            "reference": reference,
            "expected_source": expected_source,
            "trap": item.get("trap", ""),
            "context_sources": context_sources,
            "similarities": [round(chunk["similarity"], 3) for chunk in result["context"]],
            "hit_rank": hit_rank,
            # Файлы, на которые сослалась сама модель, и отказ (дословная фраза из промпта)
            "cited_sources": result["cited"],
            "is_refusal": result["answer"].strip(" «»\"") == NO_CONTEXT_ANSWER,
            "model_called": result["model_called"],
            "usage": result["usage"]
        })

    print()
    return samples


async def score_sample(sample: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Подсчёт всех метрик для одного вопроса.
    Если метрика упала с ошибкой, вместо числа сохраняется nan, а ошибка печатается.

    Args:
        sample: вопрос, ответ, найденные чанки и эталон
        metrics: метрики RAGAS по ключам из METRIC_LABELS

    Returns:
        значения метрик по тем же ключам
    """
    calls = {
        "answer_relevancy": metrics["answer_relevancy"].ascore(
            user_input=sample["user_input"],
            response=sample["response"]
        )
    }

    # Метрикам контекста нужны чанки: без них (сработал порог схожести) считать нечего
    if sample["retrieved_contexts"]:
        # Отказ судье не отдаётся: в нём нет утверждений о документах, и оценка выходит случайной.
        # Отказы считаются отдельно, сравнением с фразой отказа
        if not sample["is_refusal"]:
            calls["faithfulness"] = metrics["faithfulness"].ascore(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"]
            )
        calls["context_precision"] = metrics["context_precision"].ascore(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"]
        )

    # Метрики одного вопроса считаются параллельно
    results = await asyncio.gather(*calls.values(), return_exceptions=True)

    scores = {name: float("nan") for name in METRIC_LABELS}
    for name, res in zip(calls, results):
        if isinstance(res, BaseException):
            print(f"     [!] {name}: {type(res).__name__}: {res}")
        elif res.value is not None:
            scores[name] = float(res.value)
    return scores


async def score_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Подсчёт метрик RAGAS для всех вопросов.

    Args:
        samples: результат collect_samples

    Returns:
        значения метрик по каждому вопросу, в порядке вопросов
    """
    all_scores = []

    # Повторы и таймаут заданы явно: короткий сбой сети не должен обнулять метрику,
    # а зависший запрос - держать скрипт
    async with AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=JUDGE_MAX_RETRIES,
        timeout=JUDGE_TIMEOUT
    ) as client:
        # Модель-судья и эмбеддинги для RAGAS
        judge_llm = llm_factory(JUDGE_MODEL, client=client, max_tokens=JUDGE_MAX_TOKENS)
        judge_embeddings = embedding_factory("openai", model=JUDGE_EMBEDDING_MODEL, client=client)

        metrics = {
            "faithfulness": Faithfulness(llm=judge_llm),
            "context_precision": ContextPrecisionWithReference(llm=judge_llm),
            "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
        }

        # Вопросы идут по очереди: меньше параллельных запросов к OpenAI и виден ход оценки
        for i, sample in enumerate(samples, 1):
            print(f"  Оценка вопроса {i}/{len(samples)}...")
            all_scores.append(await score_sample(sample, metrics))

    return all_scores


def average(values: List[float]) -> float:
    """Среднее без учёта nan; если чисел нет - nan."""
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def format_score(value: float) -> str:
    """Число метрики для вывода; nan - метрику не удалось вычислить."""
    return "не удалось вычислить" if math.isnan(value) else f"{value:.4f}"


def describe_hit(sample: Dict[str, Any]) -> str:
    """Попадание в ожидаемый файл словами."""
    if not sample["expected_source"]:
        # Вопрос вне базы: правильно - не найти ничего подходящего
        return "вопрос вне базы, модель не вызывалась" if not sample["model_called"] else "вопрос вне базы, модель вызывалась"
    if sample["hit_rank"] is None:
        return f"нет: ожидался {sample['expected_source']}"
    return f"да, место {sample['hit_rank']}"


def save_results(samples: List[Dict[str, Any]], all_scores: List[Dict[str, float]]) -> str:
    """
    Сохранение результатов прогона в файл JSON: настройки, ответы, чанки и метрики.

    Args:
        samples: результат collect_samples
        all_scores: метрики по вопросам

    Returns:
        путь к сохранённому файлу
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    name = (f"{datetime.now():%Y-%m-%d-%H%M%S}-{settings['chunking_mode']}-{GROUND_TRUTH_MODE}"
            f"-topk{settings['top_k']}-ragas.json")
    path = RESULTS_DIR / name

    rows = []
    for sample, scores in zip(samples, all_scores):
        row = {key: value for key, value in sample.items() if key != "retrieved_contexts"}
        row["contexts"] = sample["retrieved_contexts"]
        # nan в JSON записывается как null
        row["scores"] = {key: (None if math.isnan(value) else round(value, 4)) for key, value in scores.items()}
        rows.append(row)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "rules": rules,
        "ground_truth_mode": GROUND_TRUTH_MODE,
        "questions_file": QUESTIONS_PATH.name,
        "judge": {"model": JUDGE_MODEL, "embedding_model": JUDGE_EMBEDDING_MODEL},
        "questions": rows
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(path)


def print_group_table(title: str, groups: Dict[str, List[int]], samples: List[Dict[str, Any]],
                      all_scores: List[Dict[str, float]]):
    """
    Вывод средних метрик по группам вопросов.

    Args:
        title: заголовок таблицы
        groups: название группы -> номера вопросов в ней
        samples: результат collect_samples
        all_scores: метрики по вопросам
    """
    print(f"\n{title}")
    print(f"   {'':<34} {'вопр.':>5} {'попад.':>7} {'Faith.':>8} {'Prec.':>8} {'Relev.':>8}")
    for name, indexes in groups.items():
        hits = sum(1 for i in indexes if samples[i]["hit_rank"] is not None)
        group_averages = [average([all_scores[i][metric] for i in indexes]) for metric in METRIC_LABELS]
        values = ["-" if math.isnan(value) else f"{value:.4f}" for value in group_averages]
        print(f"   {name[:34]:<34} {len(indexes):>5} {f'{hits}/{len(indexes)}':>7} "
              f"{values[0]:>8} {values[1]:>8} {values[2]:>8}")


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.

    Процесс:
    1. Запуск ассистента и сборка базы
    2. Генерация ответов на вопросы из файла вопросов
    3. Подсчёт метрик по каждому вопросу
    4. Вывод результатов: общие, по файлам и по каждому вопросу
    5. Сохранение результатов в папку results
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nСоздайте файл .env в папке проекта по образцу env.example:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    questions = load_questions()

    # Запуск ассистента и сборка базы
    try:
        print("[*] Инициализация RAG системы...\n")
        print(f"Настройки: модель {settings['model']}, температура {settings['temperature']}, "
              f"top_k {settings['top_k']}, бюджет {settings['context_budget']}, порог {settings['score_threshold']}")
        print(f"Документы: нарезка {settings['chunking_mode']}, "
              f"очистка {'да' if settings['clean_text'] else 'нет'}, эмбеддинги {settings['embedding_model']}")
        print(f"Эталон (ground truth): {GROUND_TRUTH_MODE}")
        print(f"Судья: {JUDGE_MODEL}, эмбеддинги {JUDGE_EMBEDDING_MODEL}\n")
        assistant = Assistant(settings, rules, log=lambda message: None)
        build = assistant.build_info
        print(f"База: чанков {build['chunks']}, чтение и нарезка {build['read_seconds']:.2f} с, "
              f"пересобрана: {'да' if build['rebuilt'] else 'нет'}")
        if build['rebuilt']:
            print(f"Векторы: из кеша {build['from_cache']}, посчитано заново {build['computed']}")
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось запустить ассистента: {e}")
        sys.exit(1)

    # Вопросы к файлам, выключенным в chunking.txt, пропускаются:
    # иначе при работе с частью базы они считались бы промахами
    active_sources = set(assistant.stats()["index"]["sources"])
    skipped = [item for item in questions if item.get("source") and item["source"] not in active_sources]
    questions = [item for item in questions if item not in skipped]
    if skipped:
        skipped_files = ", ".join(sorted({item["source"] for item in skipped}))
        print(f"[*] Пропущено вопросов: {len(skipped)} (их файлов нет в базе: {skipped_files})\n")
    if not questions:
        print("[ОШИБКА] Не осталось вопросов для оценки: проверьте поле source в файле вопросов")
        sys.exit(1)

    # Ответы и чанки по каждому вопросу
    print("=" * 70)
    samples = collect_samples(assistant, questions)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision, Answer Relevancy\n")

    # Запускаем оценку RAGAS
    try:
        all_scores = asyncio.run(score_samples(samples))
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    # Результаты сохраняются до печати: ошибка вывода на экран не должна стоить оплаченного прогона
    path = save_results(samples, all_scores)

    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)

    # Вычисляем средние значения метрик (игнорируя NaN)
    averages = {name: average([scores[name] for scores in all_scores]) for name in METRIC_LABELS}

    # Выводим общие метрики; если часть вопросов не посчитана, это видно рядом с числом
    print()
    print("[МЕТРИКИ] Средние значения:")
    for name, label in METRIC_LABELS.items():
        counted = sum(1 for scores in all_scores if not math.isnan(scores[name]))
        note = "" if counted == len(all_scores) else f"  (по {counted} из {len(all_scores)} вопросов)"
        print(f"   {label + ':':<42} {format_score(averages[name])}{note}")

    # Попадание считается по вопросам, у которых указан ожидаемый файл
    in_base = [sample for sample in samples if sample["expected_source"]]
    hits = sum(1 for sample in in_base if sample["hit_rank"] is not None)
    first = sum(1 for sample in in_base if sample["hit_rank"] == 1)
    if in_base:
        print(f"   {'Попадание в нужный файл:':<42} {hits} из {len(in_base)} (на первом месте: {first})")

    # Отказы: на вопросе вне базы отказ верен, на вопросе из базы это промах
    refusals = [i for i, sample in enumerate(samples, 1) if sample["is_refusal"]]
    missed = [i for i in refusals if samples[i - 1]["expected_source"]]
    out_of_base = sum(1 for sample in samples if not sample["expected_source"])
    print(f"   {'Отказов:':<42} {len(refusals)} (верных, на вопросах вне базы: {len(refusals) - len(missed)} "
          f"из {out_of_base}; промахов, на вопросах из базы: {len(missed)})")
    if missed:
        print(f"   {'Номера вопросов с промахом:':<42} {', '.join(map(str, missed))}")

    # Ссылка модели на источник: назван ли в ответе ожидаемый файл
    answered = [sample for sample in in_base if not sample["is_refusal"]]
    cited_right = sum(1 for sample in answered if sample["expected_source"] in sample["cited_sources"])
    cited_none = sum(1 for sample in answered if not sample["cited_sources"])
    if answered:
        print(f"   {'Ответ со ссылкой на нужный файл:':<42} {cited_right} из {len(answered)} ответов "
              f"(без ссылки: {cited_none})")

    # Вычисляем и выводим средний балл.
    # Answer Relevancy в него не входит: она занижает короткие верные ответы.
    # Если одна из двух метрик не посчитана совсем, балла нет: вторая его не заменяет
    avg_score = (averages["faithfulness"] + averages["context_precision"]) / 2
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл (Faithfulness и Context Precision): {format_score(avg_score)}")

    # Средние по файлам и по ловушкам: на каком документе нарезка сработала, а на каком нет
    by_source: Dict[str, List[int]] = {}
    by_trap: Dict[str, List[int]] = {}
    for i, sample in enumerate(samples):
        by_source.setdefault(sample["expected_source"] or "(вопрос вне базы)", []).append(i)
        if sample["trap"]:
            by_trap.setdefault(sample["trap"], []).append(i)

    print_group_table("[ПО ФАЙЛАМ]", by_source, samples, all_scores)
    if by_trap:
        print_group_table("[ПО ЛОВУШКАМ]", by_trap, samples, all_scores)

    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    for i, (sample, scores) in enumerate(zip(samples, all_scores), 1):
        print(f"\n{i}. {sample['user_input']}")
        if sample["trap"]:
            print(f"   Ловушка: {sample['trap']}")
        # Ответ ассистента: чтобы число метрики можно было сверить с тем, что оценивал судья
        print(f"   Ответ: {sample['response']}")
        print(f"   Эталон: {sample['reference']}")
        print(f"   Чанки в контексте: {', '.join(sample['context_sources']) or 'нет'}")
        print(f"   Модель сослалась на: {', '.join(sample['cited_sources']) or 'нет ссылки'}")
        print(f"   Попадание: {describe_hit(sample)}")
        for name, label in METRIC_LABELS.items():
            short_label = label.split(" (")[0]
            print(f"   {short_label + ':':<19} {format_score(scores[name])}")

    print("\n" + "=" * 70)
    print(f"[OK] Оценка завершена! Результаты сохранены: {path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
