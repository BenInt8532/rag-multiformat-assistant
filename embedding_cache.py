"""
Кеш эмбеддингов.
Хранит в SQLite векторы уже обработанных чанков: при пересборке базы
вектор считается заново только для текста, которого в кеше ещё нет.
"""

import hashlib
import sqlite3
from array import array
from datetime import datetime
from typing import Dict, List

# Сколько ключей передавать в один запрос SQLite (у него есть предел числа параметров)
SELECT_BATCH_SIZE = 500


class EmbeddingCache:
    """Кеш векторных представлений текстов."""

    def __init__(self, db_path: str):
        """
        Инициализация кеша.

        Args:
            db_path: путь к файлу базы данных SQLite
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Создание таблицы кеша, если она не существует."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def text_hash(model: str, text: str) -> str:
        """
        Ключ кеша: отпечаток текста вместе с названием модели.
        У разных моделей векторы разные, поэтому модель входит в ключ.

        Args:
            model: модель эмбеддингов
            text: текст чанка

        Returns:
            SHA-256 хеш
        """
        return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: List[str]) -> Dict[str, List[float]]:
        """
        Получение векторов из кеша.

        Args:
            model: модель эмбеддингов
            texts: тексты чанков

        Returns:
            словарь: текст -> вектор; текстов, которых в кеше нет, в словаре нет
        """
        hashes = {self.text_hash(model, text): text for text in texts}
        keys = list(hashes)
        found = {}

        conn = sqlite3.connect(self.db_path)
        for start in range(0, len(keys), SELECT_BATCH_SIZE):
            batch = keys[start:start + SELECT_BATCH_SIZE]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT text_hash, vector FROM embeddings WHERE text_hash IN ({placeholders})", batch
            )
            for text_hash, blob in rows:
                vector = array("f")
                vector.frombytes(blob)
                found[hashes[text_hash]] = vector.tolist()
        conn.close()

        return found

    def set_many(self, model: str, texts: List[str], vectors: List[List[float]]):
        """
        Сохранение векторов в кеш.

        Args:
            model: модель эмбеддингов
            texts: тексты чанков
            vectors: векторы в порядке текстов
        """
        now = datetime.now().isoformat()
        rows = [
            # Вектор хранится числами одинарной точности: втрое компактнее текста, для поиска этого достаточно
            (self.text_hash(model, text), model, array("f", vector).tobytes(), now)
            for text, vector in zip(texts, vectors)
        ]

        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (text_hash, model, vector, created_at) VALUES (?, ?, ?, ?)",
            rows
        )
        conn.commit()
        conn.close()

    def count(self) -> int:
        """Число векторов в кеше."""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.close()
        return total
