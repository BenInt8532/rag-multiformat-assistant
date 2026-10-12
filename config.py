"""
Настройки RAG ассистента.
Загружает .env и файл правил нарезки, отдаёт параметры модели, поиска и подготовки документов.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Все пути считаются от папки проекта, а не от папки, из которой запущен скрипт
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
ANSWER_CACHE_DB_PATH = BASE_DIR / "answer_cache.db"
EMBEDDING_CACHE_DB_PATH = BASE_DIR / "embedding_cache.db"
# Файл правил нарезки. Для проверки другого набора правил без правки chunking.txt
# имя файла задаётся переменной окружения CHUNKING_FILE перед запуском (в .env её не пишут)
RULES_PATH = BASE_DIR / (os.getenv("CHUNKING_FILE") or "chunking.txt")
# Файл вопросов для оценки. Другой набор из папки eval задаётся переменной QUESTIONS_FILE
QUESTIONS_PATH = BASE_DIR / "eval" / (os.getenv("QUESTIONS_FILE") or "questions.json")
RESULTS_DIR = BASE_DIR / "results"

# Режимы нарезки: rules - каждому файлу своё правило из chunking.txt,
# uniform - одно правило на все файлы (точка отсчёта для сравнения)
CHUNKING_MODES = ("rules", "uniform")

# Способы нарезки, которые умеет chunking.py, и параметры, когда правило default не задано
STRATEGIES = ("fixed", "paragraphs", "sections", "qa", "lists", "table_rows", "code")
DEFAULT_RULE = {"strategy": "paragraphs", "size": 500, "overlap": 100}


def load_env():
    """Загрузка переменных окружения из .env файла в папке проекта."""
    load_dotenv(BASE_DIR / ".env")


def _read_number(name: str, cast, default=None, minimum=None):
    """
    Чтение числа из переменной окружения с понятной ошибкой.

    Args:
        name: имя переменной
        cast: int или float
        default: значение, если переменная не задана или пустая
        minimum: наименьшее допустимое значение

    Returns:
        число или default
    """
    raw = os.getenv(name)
    # Пустое значение в .env считается незаданным
    if raw is None or not raw.strip():
        return default

    try:
        value = cast(raw.strip())
    except ValueError:
        kind = "целое число" if cast is int else "число через точку (0.3)"
        sys.exit(f"Ошибка в .env: {name} - {kind}, а не '{raw}'")

    if minimum is not None and value < minimum:
        sys.exit(f"Ошибка в .env: {name} должно быть не меньше {minimum}")
    return value


def _read_flag(name: str, default: bool) -> bool:
    """Чтение переключателя 1/0 из переменной окружения."""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "да"):
        return True
    if raw in ("0", "false", "no", "нет"):
        return False
    sys.exit(f"Ошибка в .env: {name} должно быть 1 или 0, а не '{raw}'")


def get_settings() -> Dict[str, Any]:
    """
    Чтение настроек из переменных окружения.

    Returns:
        словарь с настройками модели, поиска и подготовки документов
    """
    load_env()

    chunking_mode = (os.getenv("CHUNKING_MODE") or "rules").strip().lower()
    if chunking_mode not in CHUNKING_MODES:
        sys.exit(f"Ошибка в .env: CHUNKING_MODE должно быть rules или uniform, а не '{chunking_mode}'")

    score_threshold: Optional[float] = _read_number("SCORE_THRESHOLD", float)
    if score_threshold is not None and not 0 <= score_threshold <= 1:
        sys.exit("Ошибка в .env: SCORE_THRESHOLD должно быть от 0 до 1")

    return {
        # Модель ответа
        "model": os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        "temperature": _read_number("TEMPERATURE", float, default=0.3, minimum=0),
        # Поиск: потолок числа чанков, бюджет контекста в символах, порог схожести
        "top_k": _read_number("TOP_K", int, default=8, minimum=1),
        "context_budget": _read_number("CONTEXT_BUDGET", int, default=2000, minimum=1),
        "score_threshold": score_threshold,
        # Подготовка документов
        "embedding_model": os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
        "chunking_mode": chunking_mode,
        "clean_text": _read_flag("CLEAN_TEXT", default=True)
    }


def settings_key(settings: Dict[str, Any], names) -> str:
    """
    Строка настроек для ключа кеша.

    Args:
        settings: словарь настроек из get_settings
        names: какие настройки включить

    Returns:
        строка вида model=...|temperature=...|top_k=...
    """
    return "|".join(f"{name}={settings[name]}" for name in names)


def load_rules() -> Dict[str, Any]:
    """
    Чтение правил нарезки из chunking.txt.

    Returns:
        словарь с ключами default (правило по умолчанию), uniform (единое правило
        для режима uniform) и files (правила по именам и маскам файлов)
    """
    if not RULES_PATH.exists():
        sys.exit(f"Не найден файл правил нарезки: {RULES_PATH}")

    rules = {"default": dict(DEFAULT_RULE), "uniform": None, "files": {}}

    with open(RULES_PATH, "r", encoding="utf-8-sig") as f:
        for number, raw_line in enumerate(f, 1):
            # Всё после решётки - комментарий; строка с решёткой в начале выключена целиком
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            where = f"{RULES_PATH.name}, строка {number}"
            if ":" not in line:
                sys.exit(f"Ошибка в {where}: нужна запись вида 'имя файла : способ, параметры'")

            name, _, value = line.partition(":")
            name = name.strip()
            rule = _parse_rule(value, where)

            if name.lower() in ("default", "uniform"):
                rules[name.lower()] = rule
            else:
                rules["files"][name] = rule

    # Единая нарезка не задана - берётся правило по умолчанию
    if rules["uniform"] is None:
        rules["uniform"] = dict(rules["default"])

    # Перекрытие сверяется с размером после подстановки значений по умолчанию:
    # при перекрытии не меньше размера нарезка шла бы с шагом в один символ
    for name, rule in [("default", rules["default"]), ("uniform", rules["uniform"]), *rules["files"].items()]:
        merged = {**rules["default"], **rule}
        size, overlap = merged.get("size", 500), merged.get("overlap", 0)
        if overlap >= size:
            sys.exit(f"Ошибка в {RULES_PATH.name}, правило '{name}': перекрытие overlap={overlap} "
                     f"должно быть меньше размера size={size}")
    return rules


def _parse_rule(value: str, where: str) -> Dict[str, Any]:
    """
    Разбор правой части строки правила: "способ, параметр=значение, ...".

    Args:
        value: текст после двоеточия
        where: файл и номер строки для сообщения об ошибке

    Returns:
        правило: strategy и указанные параметры
    """
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        sys.exit(f"Ошибка в {where}: после двоеточия не указан способ нарезки")

    strategy = parts[0].lower()
    if strategy not in STRATEGIES:
        sys.exit(f"Ошибка в {where}: неизвестный способ '{parts[0]}'. Допустимые: {', '.join(STRATEGIES)}")

    rule: Dict[str, Any] = {"strategy": strategy}
    for part in parts[1:]:
        key, separator, raw = part.partition("=")
        key, raw = key.strip().lower(), raw.strip().lower()

        if not separator or not raw:
            sys.exit(f"Ошибка в {where}: параметр пишется как 'имя=значение', а не '{part}'")

        if key in ("size", "overlap"):
            if not raw.isdigit() or (key == "size" and int(raw) < 1):
                sys.exit(f"Ошибка в {where}: {key} - целое число, а не '{raw}'")
            rule[key] = int(raw)
        elif key in ("merge", "consts"):
            if raw not in ("yes", "no", "да", "нет"):
                sys.exit(f"Ошибка в {where}: {key} - это yes или no, а не '{raw}'")
            rule[key] = raw in ("yes", "да")
        else:
            sys.exit(f"Ошибка в {where}: неизвестный параметр '{key}'. Допустимые: size, overlap, merge, consts")

    return rule
