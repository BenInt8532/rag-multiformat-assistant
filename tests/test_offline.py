"""
Тесты без ключа и без сети: загрузчики, очистка, нарезка, файл правил, кеш эмбеддингов.

Запуск из папки проекта:
    python -m unittest tests.test_offline -v
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from chunking import chunk_document, find_rule  # noqa: E402
from embedding_cache import EmbeddingCache  # noqa: E402
from loaders import load_document  # noqa: E402
from prompts import split_sources  # noqa: E402

DATA_DIR = PROJECT_DIR / "data"

# Правила записаны в самом тесте: он не должен зависеть от того,
# какие строки сейчас включены в chunking.txt
RULES = {
    "default": {"strategy": "paragraphs", "size": 500, "overlap": 100},
    "uniform": {"strategy": "paragraphs", "size": 500, "overlap": 100},
    "files": {
        "glossariy.txt": {"strategy": "paragraphs", "merge": False, "overlap": 0},
        "pravila-raboty.md": {"strategy": "sections", "size": 600, "overlap": 0},
        "faq.html": {"strategy": "qa"},
        "vyplaty-i-hold.pdf": {"strategy": "paragraphs", "size": 800, "overlap": 150},
        "pervaya-kampaniya.docx": {"strategy": "lists", "size": 900},
        "offery.xlsx": {"strategy": "table_rows"},
        "stavki-po-geo.csv": {"strategy": "table_rows"},
        "tracker_ru.py": {"strategy": "code", "size": 1500},
        "tracker_en.py": {"strategy": "code", "size": 1500},
    },
}


def chunks_of(file_name: str, uniform: bool = False, clean: bool = True):
    """Тексты чанков одного файла базы."""
    document = load_document(DATA_DIR / file_name, clean=clean)
    rule = find_rule(RULES, file_name, uniform)
    return [chunk["text"] for chunk in chunk_document(document, rule, uniform)]


class LoadersTest(unittest.TestCase):
    """Тест 1: каждый загрузчик читает свой файл базы."""

    def test_all_base_files_are_read(self):
        for file_name in RULES["files"]:
            with self.subTest(file=file_name):
                document = load_document(DATA_DIR / file_name)
                self.assertTrue(document.text or document.tables, "файл прочитан пустым")

    def test_xlsx_has_two_sheets_as_tables(self):
        tables = load_document(DATA_DIR / "offery.xlsx").tables
        self.assertEqual([(len(t.header), len(t.rows)) for t in tables], [(9, 11), (5, 3)])
        self.assertIn("Архив", tables[1].name)

    def test_csv_table(self):
        table = load_document(DATA_DIR / "stavki-po-geo.csv").tables[0]
        self.assertEqual((len(table.header), len(table.rows)), (6, 10))
        self.assertEqual(table.header[0], "Оффер")

    def test_docx_keeps_step_numbers_and_table(self):
        document = load_document(DATA_DIR / "pervaya-kampaniya.docx")
        self.assertIn("1. Выберите один оффер", document.text)
        self.assertIn("10. Посчитайте ROI", document.text)
        self.assertEqual((len(document.tables[0].header), len(document.tables[0].rows)), (3, 4))


class CleaningTest(unittest.TestCase):
    """Тест 2: очистка убирает служебные части, а без очистки они остаются."""

    PDF_HEADER = "Партнёрская сеть ЛидПоток. Условия выплат и холд"

    def test_pdf_without_page_headers_and_numbers(self):
        text = load_document(DATA_DIR / "vyplaty-i-hold.pdf", clean=True).text
        self.assertNotIn(self.PDF_HEADER, text)
        self.assertNotIn("Стр. 1 из 3", text)

    def test_pdf_raw_keeps_page_headers(self):
        text = load_document(DATA_DIR / "vyplaty-i-hold.pdf", clean=False).text
        self.assertIn(self.PDF_HEADER, text)
        self.assertIn("Стр. 1 из 3", text)

    def test_html_without_menu_footer_and_sidebar(self):
        text = load_document(DATA_DIR / "faq.html", clean=True).text
        for junk in ("Главная", "Все права защищены", "Акция месяца", "openChat"):
            self.assertNotIn(junk, text)
        self.assertIn("Как настроить постбэк?", text)

    def test_html_raw_keeps_menu(self):
        text = load_document(DATA_DIR / "faq.html", clean=False).text
        self.assertIn("Главная", text)
        self.assertIn("Все права защищены", text)


class ChunkingByRulesTest(unittest.TestCase):
    """Тест 3: нарезка по правилам даёт чанки, понятные без соседей."""

    def test_glossary_one_term_per_chunk(self):
        chunks = chunks_of("glossariy.txt")
        self.assertEqual(len(chunks), 23)
        self.assertIn("Холд - срок проверки лидов.", chunks)

    def test_faq_question_stays_with_answer(self):
        pairs = [chunk for chunk in chunks_of("faq.html") if chunk.split("\n")[0].endswith("?")]
        self.assertEqual(len(pairs), 9)
        for pair in pairs:
            self.assertGreaterEqual(len(pair.split("\n")), 2, f"вопрос без ответа: {pair}")
        postback = next(pair for pair in pairs if pair.startswith("Как настроить постбэк?"))
        self.assertIn("Отправить тестовый постбэк", postback)

    def test_table_rows_carry_column_names(self):
        csv_chunks = chunks_of("stavki-po-geo.csv")
        self.assertEqual(len(csv_chunks), 10)
        for chunk in csv_chunks:
            self.assertIn("Оффер:", chunk)
            self.assertIn("Ставка за лид, руб:", chunk)

        xlsx_chunks = chunks_of("offery.xlsx")
        self.assertEqual(len(xlsx_chunks), 14)
        self.assertEqual(sum("лист Архив" in chunk for chunk in xlsx_chunks), 3)

    def test_code_chunks_are_whole_functions_with_label(self):
        chunks = chunks_of("tracker_ru.py")
        for chunk in chunks:
            self.assertTrue(chunk.startswith("Файл tracker_ru.py, "), chunk[:60])
        roi = next(chunk for chunk in chunks if "функция calc_roi" in chunk.split("\n")[0])
        self.assertIn("def calc_roi(revenue, cost):", roi)
        self.assertIn("return (revenue - cost) / cost * 100", roi)
        self.assertIn("ROI отвечает на главный вопрос", roi, "комментарий над функцией не попал в её чанк")

    def test_code_consts_puts_used_settings_into_function_chunk(self):
        document = load_document(DATA_DIR / "tracker_en.py", clean=True)
        rule = {"strategy": "code", "size": 1500, "overlap": 0, "consts": True}
        chunks = [chunk["text"] for chunk in chunk_document(document, rule, uniform=False)]
        pause = next(chunk for chunk in chunks if "функция find_groups_to_pause" in chunk)
        self.assertIn("MIN_VISITS = 150", pause)
        self.assertIn("STOP_MARGIN = -35.0", pause)
        # Функция, которая настройками не пользуется, остаётся без приписки
        margin = next(chunk for chunk in chunks if "функция margin" in chunk)
        self.assertNotIn("Настройки из начала файла", margin)
        # Без параметра нарезка прежняя
        plain = [chunk["text"] for chunk in chunk_document(document, {**rule, "consts": False}, uniform=False)]
        self.assertNotIn("MIN_VISITS = 150", next(c for c in plain if "функция find_groups_to_pause" in c))

    def test_section_title_is_in_its_chunk(self):
        chunks = chunks_of("pravila-raboty.md")
        banned = next(chunk for chunk in chunks if "Мотивированный трафик" in chunk)
        self.assertTrue(banned.startswith("Запрещённые источники трафика"), banned[:80])

    def test_numbered_list_is_not_split(self):
        chunks = chunks_of("pervaya-kampaniya.docx")
        steps = next(chunk for chunk in chunks if "1. Выберите один оффер" in chunk)
        self.assertIn("10. Посчитайте ROI", steps)
        self.assertIn("Выполняйте шаги строго по порядку", steps, "вводная фраза оторвана от списка")

    def test_pdf_chunks_do_not_start_mid_word(self):
        for chunk in chunks_of("vyplaty-i-hold.pdf"):
            self.assertFalse(chunk[0].islower(), f"чанк начинается с обрывка: {chunk[:60]}")


class RulesFileTest(unittest.TestCase):
    """Тест 5: файл правил chunking.txt."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.original_path = config.RULES_PATH
        config.RULES_PATH = Path(self.folder.name) / "chunking.txt"

    def tearDown(self):
        config.RULES_PATH = self.original_path
        self.folder.cleanup()

    def write_rules(self, text: str):
        config.RULES_PATH.write_text(text, encoding="utf-8")

    def test_commented_line_switches_file_off(self):
        self.write_rules("default: paragraphs, size=400\nfaq.html: qa\n# glossariy.txt: paragraphs, merge=no\n")
        rules = config.load_rules()
        self.assertTrue(find_rule(rules, "faq.html", uniform=False)["use"])
        self.assertFalse(find_rule(rules, "glossariy.txt", uniform=False)["use"])
        # В режиме uniform состав базы тот же, меняется только способ
        self.assertFalse(find_rule(rules, "glossariy.txt", uniform=True)["use"])
        self.assertEqual(find_rule(rules, "faq.html", uniform=True)["strategy"], "paragraphs")

    def test_parameters_and_defaults(self):
        self.write_rules("default: paragraphs, size=400, overlap=50\nrules.md: sections, size=700\n")
        rule = find_rule(config.load_rules(), "rules.md", uniform=False)
        self.assertEqual((rule["strategy"], rule["size"], rule["overlap"]), ("sections", 700, 50))

    def test_typo_gives_message_with_line_number(self):
        self.write_rules("default: paragraphs\nfaq.html: qaa\n")
        with self.assertRaises(SystemExit) as error:
            config.load_rules()
        self.assertIn("строка 2", str(error.exception))

    def test_overlap_not_less_than_size_is_error(self):
        # Перекрытие 100 приходит из default: при size=100 шаг нарезки стал бы равен одному символу
        self.write_rules("default: paragraphs, size=500, overlap=100\nnotes.txt: fixed, size=100\n")
        with self.assertRaises(SystemExit) as error:
            config.load_rules()
        self.assertIn("notes.txt", str(error.exception))


class SourcesLineTest(unittest.TestCase):
    """Строка «Источники» в конце ответа модели: номера фрагментов превращаются в имена файлов."""

    CHUNKS = [{"source": "faq.html"}, {"source": "offery.xlsx"}, {"source": "faq.html"}]

    def test_numbers_become_file_names(self):
        text, files = split_sources("Холд длится 21 день.\nИсточники: [1], [3], [2]", self.CHUNKS)
        self.assertEqual(text, "Холд длится 21 день.")
        self.assertEqual(files, ["faq.html", "offery.xlsx"])

    def test_sources_in_the_same_line_with_dot(self):
        # Так модель писала в прогоне: без перевода строки и с точкой в конце
        text, files = split_sources("Холд длится 21 день. Источники: [1], [2].", self.CHUNKS)
        self.assertEqual((text, files), ("Холд длится 21 день.", ["faq.html", "offery.xlsx"]))

    def test_answer_without_line_is_kept(self):
        text, files = split_sources("В документах нет информации по этому вопросу.", self.CHUNKS)
        self.assertEqual((text, files), ("В документах нет информации по этому вопросу.", []))

    def test_unknown_number_is_skipped(self):
        text, files = split_sources("Ответ.\n\nИсточник: [7], [2]", self.CHUNKS)
        self.assertEqual((text, files), ("Ответ.", ["offery.xlsx"]))


class EmbeddingCacheTest(unittest.TestCase):
    """Тест 6: при втором сборе базы векторы заново не считаются."""

    def test_second_build_requests_nothing(self):
        from index import ChunkIndex

        requested = []

        def fake_request_vectors(texts):
            requested.append(list(texts))
            return [[float(len(text)), 0.5, -1.0] for text in texts]

        with tempfile.TemporaryDirectory() as folder:
            index = types.SimpleNamespace(
                embedding_cache=EmbeddingCache(str(Path(folder) / "cache.db")),
                embedding_model="test-model",
                _request_vectors=fake_request_vectors,
                log=lambda message: None,
            )
            texts = chunks_of("glossariy.txt") + chunks_of("stavki-po-geo.csv")

            first, first_cached, first_computed = ChunkIndex._vectors_for(index, texts)
            second, second_cached, second_computed = ChunkIndex._vectors_for(index, texts)
            # Добавился один новый чанк: считается только он
            third, third_cached, third_computed = ChunkIndex._vectors_for(index, texts + ["новый чанк"])

        self.assertEqual(len(requested), 2, "при втором сборе были запросы за векторами")
        self.assertEqual((first_cached, first_computed), (0, len(texts)))
        self.assertEqual((second_cached, second_computed), (len(texts), 0))
        self.assertEqual((third_cached, third_computed), (len(texts), 1))
        self.assertEqual(requested[1], ["новый чанк"])
        self.assertEqual(first, second)
        self.assertEqual(len(third), len(texts) + 1)


class AnswerCacheTest(unittest.TestCase):
    """Тест 6: кеш ответов находит ответ только в своей области настроек."""

    def test_scope_and_normalization(self):
        from answer_cache import AnswerCache

        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "answers.db")
            cache = AnswerCache(path, scope="top_k=3|docs=aaa")
            cache.put("Что такое RAG?", "ответ", [{"source": "glossariy.txt", "text": "..."}])

            # Регистр и лишние пробелы не мешают, пропущенный знак препинания мешает
            self.assertEqual(cache.get("что  такое rag?")["answer"], "ответ")
            self.assertIsNone(cache.get("Что такое RAG"))
            self.assertEqual(cache.get("Что такое RAG?")["sources"][0]["source"], "glossariy.txt")

            # Другие настройки или другая база - другая область: ответа там нет
            self.assertIsNone(AnswerCache(path, scope="top_k=7|docs=aaa").get("Что такое RAG?"))
            self.assertIsNone(AnswerCache(path, scope="top_k=3|docs=bbb").get("Что такое RAG?"))

            self.assertEqual(cache.stats()["entries"], 1)
            cache.clear()
            self.assertEqual(cache.stats()["entries"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
