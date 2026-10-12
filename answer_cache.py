"""
Кеш ответов ассистента.

Готовый ответ хранится в SQLite под ключом из вопроса и "области": строки с настройками
и отпечатком базы знаний. Сменились настройки или документы - область другая,
и старый ответ не находится.
"""

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, List, Optional


def normalize_question(question: str) -> str:
    """Вопрос в едином виде: нижний регистр, пробелы по одному. Знаки препинания не трогаются."""
    return " ".join(question.lower().split())


class AnswerCache:
    """Кеш "вопрос - ответ" в файле SQLite."""

    def __init__(self, db_path: str, scope: str = ""):
        """
        Args:
            db_path: путь к файлу базы данных
            scope: область действия записей: настройки ответа и отпечаток базы знаний
        """
        self.db_path = db_path
        self.scope = scope

        with closing(self._connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS answers (
                    key TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _key(self, question: str) -> str:
        """Ключ записи: отпечаток области и вопроса в едином виде."""
        return hashlib.sha256(f"{self.scope}\n{normalize_question(question)}".encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Поиск готового ответа.

        Args:
            question: вопрос пользователя

        Returns:
            словарь с ответом, источниками и временем записи либо None, если ответа в кеше нет
        """
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT answer, sources, saved_at FROM answers WHERE key = ?", (self._key(question),)
            ).fetchone()

        if row is None:
            return None
        return {"answer": row[0], "sources": json.loads(row[1]), "saved_at": row[2]}

    def put(self, question: str, answer: str, sources: List[Dict[str, str]]):
        """
        Сохранение ответа. Повторная запись того же вопроса заменяет прежнюю.

        Args:
            question: вопрос пользователя
            answer: ответ ассистента
            sources: чанки, по которым дан ответ: файл и текст
        """
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO answers (key, question, answer, sources, saved_at) VALUES (?, ?, ?, ?, ?)",
                (self._key(question), question, answer, json.dumps(sources, ensure_ascii=False),
                 datetime.now().isoformat(timespec="seconds"))
            )

    def clear(self):
        """Удаление всех записей, для любых настроек."""
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM answers")

    def stats(self) -> Dict[str, Any]:
        """
        Сведения о кеше.

        Returns:
            число записей, размер файла в мегабайтах, время первой и последней записи
        """
        with closing(self._connect()) as connection:
            entries, first_saved, last_saved = connection.execute(
                "SELECT COUNT(*), MIN(saved_at), MAX(saved_at) FROM answers"
            ).fetchone()

        size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {
            "entries": entries,
            "size_mb": size / (1024 * 1024),
            "first_saved": first_saved,
            "last_saved": last_saved
        }
