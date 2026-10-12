"""
Способы нарезки документов на чанки.

Способ и его параметры для каждого файла задаются правилом из chunking.txt:
- fixed      - по числу символов, не глядя на смысл (точка отсчёта для сравнения);
- paragraphs - по абзацам, длинный абзац по предложениям, с перекрытием;
- sections   - по разделам: заголовок приклеивается к каждому чанку своего раздела;
- qa         - пара "вопрос и ответ" - один чанк;
- lists      - как paragraphs, но список с вводной фразой не разрывается;
- table_rows - строка таблицы - один чанк с названиями столбцов;
- code       - функция или класс - один чанк с подписью "файл, функция".
"""

import ast
import fnmatch
import re
from typing import Any, Dict, List, Tuple

from loaders import LIST_ITEM, Document, Table

# Граница предложений: пробел или перенос строки после знака конца предложения
SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")
# Границы блока с примером кода в Markdown и служебная пометка строки внутри такого блока
CODE_FENCES = ("```", "~~~")
CODE_LINE_MARK = "\x00"
# Наибольшая длина чанка в символах: с запасом ниже предела модели эмбеддингов (около 8000 токенов)
MAX_CHUNK_LENGTH = 6000
QUESTION_PREFIX = re.compile(r"^(вопрос|question|в|q)\s*[:.]\s*", re.IGNORECASE)


def find_rule(rules: Dict[str, Any], file_name: str, uniform: bool) -> Dict[str, Any]:
    """
    Выбор правила нарезки для файла.

    Args:
        rules: правила из chunking.txt
        file_name: имя файла
        uniform: одно правило на все файлы (режим uniform)

    Returns:
        правило: strategy и параметры; недостающие параметры берутся из правила default.
        Параметр use говорит, включён ли файл в базу: файл включён,
        только если в chunking.txt у него есть строка (без решётки)
    """
    # Берётся первое подходящее правило: точные имена записывают выше масок
    file_rule = next(
        (value for pattern, value in rules["files"].items()
         if fnmatch.fnmatch(file_name.lower(), pattern.lower())),
        None
    )

    # Нарезка в режиме uniform единая, но состав базы всё равно задают строки файлов
    chosen = rules["uniform"] if uniform else (file_rule or {})
    return {**rules["default"], **chosen, "use": file_rule is not None}


def chunk_document(document: Document, rule: Dict[str, Any], uniform: bool) -> List[Dict[str, str]]:
    """
    Нарезка документа на чанки по правилу.

    Args:
        document: документ в общем виде (текст и таблицы)
        rule: правило нарезки для этого файла
        uniform: единая нарезка - таблицы превращаются в текст и режутся вместе с ним

    Returns:
        список чанков: text - текст чанка, strategy - способ, которым он получен
    """
    strategy = rule["strategy"]
    size = int(rule.get("size", 500))
    overlap = int(rule.get("overlap", 0))
    text = document.text

    if uniform:
        # Таблица как обычный текст: названия столбцов окажутся только в первом её чанке
        tables_text = "\n\n".join(table_to_text(table) for table in document.tables)
        text = "\n\n".join(part for part in (text, tables_text) if part)
        tables = []
    else:
        tables = document.tables

    if strategy == "fixed":
        texts = chunk_fixed(text, size, overlap)
    elif strategy == "sections":
        texts = chunk_sections(text, size, overlap)
    elif strategy == "qa":
        texts = chunk_qa(text)
    elif strategy == "lists":
        texts = chunk_lists(text, size)
    elif strategy == "code":
        texts = chunk_code(text, document.name, size, overlap, consts=bool(rule.get("consts", False)))
    else:
        # paragraphs, а также текст рядом с таблицами в правиле table_rows
        texts = chunk_paragraphs(text, size, overlap, merge=bool(rule.get("merge", True)))

    chunks = [{"text": chunk, "strategy": strategy if strategy != "table_rows" else "paragraphs"} for chunk in texts]

    # Таблицы в любом правиле, кроме единой нарезки, режутся по строкам
    for table in tables:
        chunks.extend({"text": chunk, "strategy": "table_rows"} for chunk in chunk_table_rows(table))

    # Способы, которые режут по смыслу (вопрос-ответ, список, функция, строка таблицы), не ограничивают
    # длину чанка. Слишком длинный чанк модель эмбеддингов не примет: он режется по словам
    limited = []
    for chunk in chunks:
        if len(chunk["text"]) > MAX_CHUNK_LENGTH:
            limited.extend(
                {"text": part, "strategy": chunk["strategy"]}
                for part in _wrap_by_words(chunk["text"], MAX_CHUNK_LENGTH)
            )
        else:
            limited.append(chunk)

    return limited


def chunk_fixed(text: str, size: int, overlap: int) -> List[str]:
    """
    Нарезка по числу символов, без учёта слов и предложений.

    Args:
        text: исходный текст
        size: размер чанка в символах
        overlap: размер перекрытия между чанками

    Returns:
        список чанков
    """
    step = max(size - overlap, 1)
    chunks = [text[start:start + size].strip() for start in range(0, len(text), step)]
    return [chunk for chunk in chunks if chunk]


def chunk_paragraphs(text: str, size: int, overlap: int, merge: bool = True) -> List[str]:
    """
    Нарезка по абзацам.

    Соседние абзацы собираются в один чанк, пока помещаются в size. Абзац длиннее size
    делится по предложениям. Новый чанк начинается с хвоста предыдущего (перекрытие):
    хвост добавляется сверх size, поэтому чанк может выйти длиннее size на величину перекрытия.

    Args:
        text: исходный текст, абзацы разделены пустой строкой
        size: целевой размер чанка в символах
        overlap: размер перекрытия между чанками в символах
        merge: собирать ли соседние абзацы в один чанк; False - каждый абзац отдельным чанком

    Returns:
        список чанков
    """
    chunks = []
    current = ""

    for paragraph in (part.strip() for part in text.split("\n\n")):
        if not paragraph:
            continue

        if len(paragraph) > size:
            # Длинный абзац не смешивается с соседями: закрываем начатый чанк и делим абзац
            if current:
                chunks.append(current)
            pieces = pack_sentences(paragraph, size, overlap)
            chunks.extend(pieces[:-1])
            current = pieces[-1]
        elif merge and current and len(current) + 2 + len(paragraph) <= size:
            current = f"{current}\n\n{paragraph}"
        else:
            tail = ""
            if current:
                chunks.append(current)
                tail = tail_sentences(current, overlap)
            current = f"{tail}\n\n{paragraph}" if tail else paragraph

    if current:
        chunks.append(current)

    return chunks


def split_sentences(text: str) -> List[str]:
    """
    Деление текста на предложения.
    Границей считается пробел или перенос строки после точки, восклицательного
    или вопросительного знака и многоточия.
    """
    return [sentence.strip() for sentence in SENTENCE_BREAK.split(text) if sentence.strip()]


def tail_sentences(chunk: str, overlap: int) -> str:
    """
    Хвост чанка для перекрытия со следующим чанком: целые предложения его последнего
    абзаца, взятые с конца, пока вместе они не длиннее overlap.

    Предложение никогда не режется. Если уже последнее предложение длиннее overlap,
    оно берётся целиком при длине до двух overlap, иначе перекрытия нет.

    Args:
        chunk: текст закрытого чанка
        overlap: размер перекрытия в символах

    Returns:
        текст перекрытия или пустая строка
    """
    if overlap <= 0:
        return ""

    sentences = split_sentences(chunk.rsplit("\n\n", 1)[-1])
    taken = []
    for sentence in reversed(sentences):
        if len(" ".join([sentence] + taken)) > overlap:
            break
        taken.insert(0, sentence)

    if not taken and sentences and len(sentences[-1]) <= overlap * 2:
        taken = [sentences[-1]]

    return " ".join(taken)


def pack_sentences(paragraph: str, size: int, overlap: int) -> List[str]:
    """
    Деление абзаца, который длиннее size: предложения набираются в куски до size.
    Предложение длиннее size (строка таблицы, перечень без точек) режется по словам.

    Args:
        paragraph: текст абзаца
        size: целевой размер куска в символах
        overlap: размер перекрытия между соседними кусками

    Returns:
        список кусков
    """
    units = []
    for sentence in split_sentences(paragraph):
        units.extend(_wrap_by_words(sentence, size) if len(sentence) > size else [sentence])

    pieces = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > size:
            pieces.append(current)
            tail = tail_sentences(current, overlap)
            current = f"{tail} {unit}" if tail else unit
        else:
            current = f"{current} {unit}" if current else unit

    if current:
        pieces.append(current)

    return pieces


def _wrap_by_words(text: str, size: int) -> List[str]:
    """
    Разбиение текста на куски не длиннее size по границам слов.

    Args:
        text: текст без деления на предложения
        size: наибольшая длина куска

    Returns:
        список кусков
    """
    pieces = []
    current = ""

    for word in text.split():
        if current and len(current) + len(word) + 1 > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word

    if current:
        pieces.append(current)

    return pieces


def is_heading(paragraph: str, allow_plain: bool = True) -> bool:
    """
    Проверка, является ли абзац заголовком.

    Заголовок - абзац с меткой "# " (её ставят загрузчики) либо короткая строка
    без знака препинания в конце, которая не является пунктом списка.

    Args:
        paragraph: абзац текста
        allow_plain: считать ли заголовком короткую строку без метки. Выключается для
            документов, где заголовки размечены: там короткая строка - обычный текст
    """
    if paragraph.startswith("#"):
        return True
    if not allow_plain:
        return False
    return (
        "\n" not in paragraph
        and len(paragraph) <= 80
        and paragraph[-1] not in ".,;:!?"
        and not LIST_ITEM.match(paragraph)
    )


def split_sections(text: str) -> List[Tuple[str, str]]:
    """
    Разбиение текста на разделы.

    Args:
        text: исходный текст

    Returns:
        список пар (заголовок, текст раздела); у текста до первого заголовка заголовок пустой
    """
    sections = []
    title = ""
    body = []

    # В блоке с примером кода (между строками ``` или ~~~) решётка - комментарий, а не заголовок:
    # такие строки помечаются, чтобы разбор заголовков их не трогал
    protected = []
    in_code_block = False
    for line in text.split("\n"):
        if line.lstrip().startswith(CODE_FENCES):
            in_code_block = not in_code_block
        elif in_code_block and line.lstrip().startswith("#"):
            line = CODE_LINE_MARK + line
        protected.append(line)
    text = "\n".join(protected)

    # Заголовок Markdown может стоять вплотную к тексту или к другому заголовку:
    # каждая строка с "#" в начале выделяется в отдельный абзац
    text = re.sub(r"(?m)^(#{1,6}\s.*)$", r"\n\n\1\n\n", text)

    # Если заголовки в документе размечены, заголовком считается только размеченная строка
    has_marked_headings = bool(re.search(r"(?m)^#{1,6}\s", text))

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if is_heading(paragraph, allow_plain=not has_marked_headings):
            if body:
                sections.append((title, "\n\n".join(body)))
                title = ""
            # В Markdown текст может идти сразу под заголовком, без пустой строки:
            # заголовок - только первая строка абзаца, остальное - текст раздела
            heading_line, _, rest = paragraph.partition("\n")
            new_title = heading_line.lstrip("#").strip()
            # Заголовок без текста под ним - название части: добавляется к следующему заголовку
            title = f"{title} / {new_title}" if title else new_title
            body = [rest.strip()] if rest.strip() else []
        else:
            body.append(paragraph)

    if body:
        sections.append((title, "\n\n".join(body)))
    elif title:
        # Последняя короткая строка документа принята за заголовок, а текста под ней нет:
        # она остаётся в базе обычным текстом, а не теряется
        sections.append(("", title))

    # Пометки строк кода убираются: в чанки текст идёт в исходном виде
    return [(title, body.replace(CODE_LINE_MARK, "")) for title, body in sections]


def chunk_sections(text: str, size: int, overlap: int) -> List[str]:
    """
    Нарезка по разделам: раздел целиком, а длинный раздел - по абзацам.
    Заголовок раздела стоит в начале каждого его чанка.

    Args:
        text: исходный текст
        size: целевой размер чанка в символах
        overlap: размер перекрытия между чанками одного раздела

    Returns:
        список чанков
    """
    chunks = []

    for title, body in split_sections(text):
        if not title:
            chunks.extend(chunk_paragraphs(body, size, overlap))
        elif len(title) + len(body) + 2 <= size:
            chunks.append(f"{title}\n\n{body}")
        else:
            # Место под заголовок вычитается из размера чанка
            body_size = max(size - len(title) - 2, 100)
            chunks.extend(
                f"{title}\n\n{part}"
                for part in chunk_paragraphs(body, body_size, overlap)
            )

    return chunks


def is_question(line: str) -> bool:
    """Проверка, является ли строка вопросом: кончается на "?" или начинается с "Вопрос:"."""
    line = line.lstrip("#").strip()
    return line.endswith("?") or bool(QUESTION_PREFIX.match(line))


def chunk_qa(text: str) -> List[str]:
    """
    Нарезка по парам "вопрос и ответ": вопрос и весь текст до следующего вопроса - один чанк.
    Текст до первого вопроса идёт отдельным чанком.

    Args:
        text: исходный текст

    Returns:
        список чанков
    """
    chunks = []
    current = []
    has_question = False

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if is_question(line):
            # Одинокий заголовок страницы до первого вопроса чанком не становится
            if current and (has_question or len(current) > 1):
                chunks.append("\n".join(current))
            current = [line.lstrip("#").strip()]
            has_question = True
        else:
            current.append(line.lstrip("#").strip())

    if current:
        chunks.append("\n".join(current))

    return chunks


def chunk_lists(text: str, size: int) -> List[str]:
    """
    Нарезка по абзацам, при которой список не разрывается.
    Вводная фраза с двоеточием остаётся в одном чанке со своим списком.

    Args:
        text: исходный текст
        size: целевой размер чанка в символах; список длиннее size идёт отдельным чанком целиком

    Returns:
        список чанков
    """
    # Блоки: обычные абзацы и списки (несколько абзацев-пунктов подряд - один блок)
    blocks = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Список - абзац, в котором пунктами является не меньше половины строк:
        # так распознаётся и список, идущий сразу под вводной фразой, без пустой строки
        lines = paragraph.split("\n")
        is_list = sum(1 for line in lines if LIST_ITEM.match(line)) * 2 >= len(lines)
        if is_list and blocks and (blocks[-1]["is_list"] or blocks[-1]["text"].endswith(":")):
            blocks[-1]["text"] += "\n" + paragraph
            blocks[-1]["is_list"] = True
        else:
            blocks.append({"text": paragraph, "is_list": is_list})

    chunks = []
    current_chunk = ""

    for block in blocks:
        if current_chunk and len(current_chunk) + len(block["text"]) + 2 <= size:
            current_chunk += "\n\n" + block["text"]
            continue

        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        if len(block["text"]) > size and not block["is_list"]:
            # Длинный обычный абзац разбивается по предложениям
            sentence_chunks = pack_sentences(block["text"], size, 0)
            chunks.extend(sentence_chunks[:-1])
            current_chunk = sentence_chunks[-1]
        else:
            current_chunk = block["text"]

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_table_rows(table: Table) -> List[str]:
    """
    Нарезка таблицы по строкам: каждая строка - отдельный чанк вида
    "Таблица «название». столбец: значение; столбец: значение".

    Args:
        table: таблица документа

    Returns:
        список чанков
    """
    chunks = []
    for row in table.rows:
        pairs = [f"{column}: {value}" for column, value in zip(table.header, row) if value]
        if pairs:
            chunks.append(f"Таблица «{table.name}». " + "; ".join(pairs))
    return chunks


def table_to_text(table: Table) -> str:
    """Таблица как обычный текст: название, строка заголовков и строки значений."""
    lines = [table.name, " | ".join(table.header)]
    lines.extend(" | ".join(row) for row in table.rows)
    # Каждая строка таблицы - отдельный абзац: так её видит нарезка по абзацам
    return "\n\n".join(lines)


def chunk_code(text: str, file_name: str, size: int, overlap: int, consts: bool = False) -> List[str]:
    """
    Нарезка кода Python: функция или класс - один чанк. Комментарии над функцией
    и код между функциями идут вместе со следующей функцией. Начало файла (описание,
    импорты, настройки) - отдельный чанк. Большой класс режется по методам.

    Args:
        text: исходный код
        file_name: имя файла, идёт в подпись каждого чанка
        size: класс длиннее size режется по методам
        overlap: перекрытие для запасной нарезки, если код не разбирается
        consts: дописывать к чанку функции настройки из начала файла, которые она использует.
            Без этого число порога лежит в чанке «начало файла», а в функции стоит только его имя

    Returns:
        список чанков с подписью "Файл ..., функция ..."
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Не Python или код с ошибкой: режем как обычный текст
        return chunk_paragraphs(text, size, overlap)

    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    nodes = [node for node in tree.body if isinstance(node, definitions)]
    if not nodes:
        return chunk_paragraphs(text, size, overlap)

    lines = text.split("\n")
    chunks = []

    def source(start: int, end: int) -> str:
        """Строки кода с номера start по номер end (включительно), без пустых краёв."""
        return "\n".join(lines[start - 1:end]).strip("\n")

    def first_line(node: ast.AST, previous_end: int) -> int:
        """Первая строка определения вместе с декораторами и комментариями над ним."""
        start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        while start - 1 > previous_end and lines[start - 2].strip().startswith("#"):
            start -= 1
        return start

    # Шапка файла: всё до первой функции или класса
    header = source(1, first_line(nodes[0], 0) - 1)
    if header.strip():
        chunks.append(f"Файл {file_name}, начало файла: описание, импорты, настройки\n\n{header}")

    # Настройки файла: простые присваивания на верхнем уровне (имя -> строки кода)
    settings: Dict[str, str] = {}
    if consts:
        for item in tree.body:
            if isinstance(item, ast.Assign):
                names = [target.id for target in item.targets if isinstance(target, ast.Name)]
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                names = [item.target.id]
            else:
                continue
            for name in names:
                settings[name] = source(item.lineno, item.end_lineno)

    previous_end = first_line(nodes[0], 0) - 1
    for node in nodes:
        # Код между определениями (настройки, комментарии) идёт вместе со следующим определением
        start = previous_end + 1
        code = source(start, node.end_lineno)
        previous_end = node.end_lineno

        if isinstance(node, ast.ClassDef) and len(code) > size:
            chunks.extend(_chunk_class(node, lines, file_name, start))
        else:
            kind = "класс" if isinstance(node, ast.ClassDef) else "функция"
            # Настройки, имена которых встречаются в теле, в порядке их записи в файле
            used = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            used_settings = [line for name, line in settings.items() if name in used and line not in code]
            if used_settings:
                code += (f"\n\n# Настройки из начала файла, которые использует {kind}:\n"
                         + "\n".join(dict.fromkeys(used_settings)))
            chunks.append(f"Файл {file_name}, {kind} {node.name}\n\n{code}")

    # Код после последнего определения: запуск скрипта
    tail = source(previous_end + 1, len(lines))
    if tail.strip():
        chunks.append(f"Файл {file_name}, код запуска в конце файла\n\n{tail}")

    return [part for chunk in chunks for part in _split_long_code_chunk(chunk)]


def _split_long_code_chunk(chunk: str) -> List[str]:
    """
    Разбиение слишком длинного чанка кода на части по строкам кода.
    Переносы строк сохраняются, подпись повторяется в каждой части с её номером.

    Args:
        chunk: чанк вида "подпись\\n\\nкод"

    Returns:
        исходный чанк, если он не длиннее MAX_CHUNK_LENGTH, иначе его части
    """
    if len(chunk) <= MAX_CHUNK_LENGTH:
        return [chunk]

    label, _, code = chunk.partition("\n\n")
    # Запас под подпись с номером части
    limit = MAX_CHUNK_LENGTH - len(label) - 40

    parts = []
    current = []
    current_length = 0
    for line in code.split("\n"):
        if current and current_length + len(line) + 1 > limit:
            parts.append("\n".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += len(line) + 1
    if current:
        parts.append("\n".join(current))

    return [f"{label}, часть {i} из {len(parts)}\n\n{part}" for i, part in enumerate(parts, 1)]


def _chunk_class(node: ast.ClassDef, lines: List[str], file_name: str, start: int) -> List[str]:
    """
    Нарезка большого класса: описание класса и каждый метод - отдельные чанки.
    Весь код класса распределяется по чанкам без остатка: код перед классом, декораторы
    и поля идут в чанк описания, а код между методами - в чанк предыдущего метода.

    Args:
        node: класс из разбора кода
        lines: строки файла
        file_name: имя файла для подписи
        start: первая строка куска кода, который относится к классу (с кодом перед ним)

    Returns:
        список чанков
    """
    def source(first: int, last: int) -> str:
        """Строки кода с номера first по номер last (включительно), без пустых краёв."""
        return "\n".join(lines[first - 1:last]).strip("\n")

    methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not methods:
        return [f"Файл {file_name}, класс {node.name}\n\n{source(start, node.end_lineno)}"]

    # Первая строка каждого метода вместе с декораторами и комментариями над ним
    method_starts = []
    previous_end = node.lineno
    for method in methods:
        first = min([method.lineno] + [decorator.lineno for decorator in method.decorator_list])
        while first - 1 > previous_end and lines[first - 2].strip().startswith("#"):
            first -= 1
        method_starts.append(first)
        previous_end = method.end_lineno

    chunks = []

    # Строка class, описание класса и его поля до первого метода
    class_header = source(start, method_starts[0] - 1)
    if class_header.strip():
        chunks.append(f"Файл {file_name}, класс {node.name}: описание\n\n{class_header}")

    for i, method in enumerate(methods):
        # Чанк метода тянется до начала следующего метода, а у последнего - до конца класса
        last = method_starts[i + 1] - 1 if i + 1 < len(methods) else node.end_lineno
        chunks.append(
            f"Файл {file_name}, класс {node.name}, метод {method.name}\n\n{source(method_starts[i], last)}"
        )

    return chunks
