"""
Индекс базы знаний: чанки документов с векторами в ChromaDB.

Индекс собирается из папки с документами по правилам нарезки. Что уже посчитано, заново
не считается: векторы берутся из кеша эмбеддингов, а база не трогается, пока не изменились
документы или правила.
"""

import hashlib
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

import chromadb
from chromadb.errors import NotFoundError
from openai import OpenAI

from chunking import chunk_document, find_rule
from embedding_cache import EmbeddingCache
from loaders import list_documents, load_document

# Сколько текстов отправлять в OpenAI одним запросом на эмбеддинги
EMBEDDING_BATCH_SIZE = 64
# Сколько чанков добавлять в ChromaDB одним вызовом: у неё есть предел размера порции
ADD_BATCH_SIZE = 1000


def collection_name_for(settings: Dict[str, Any]) -> str:
    """
    Имя коллекции для набора настроек подготовки документов.
    У каждой нарезки, режима очистки и модели эмбеддингов своя коллекция:
    переключение между ними не пересобирает уже готовую базу.

    Args:
        settings: настройки из config.get_settings

    Returns:
        имя, допустимое для ChromaDB
    """
    cleaning = "clean" if settings["clean_text"] else "raw"
    name = f"docs-{settings['chunking_mode']}-{cleaning}-{settings['embedding_model']}"
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)[:63].strip("-_")


class ChunkIndex:
    """Чанки базы знаний и поиск по ним."""

    def __init__(self, name: str, directory: Path, embedding_model: str, api_key: str,
                 embedding_cache: EmbeddingCache, log: Callable[[str], None] = print):
        """
        Args:
            name: имя коллекции в ChromaDB
            directory: папка, где ChromaDB хранит данные
            embedding_model: модель OpenAI для эмбеддингов
            api_key: ключ OpenAI
            embedding_cache: кеш эмбеддингов
            log: куда писать ход работы
        """
        self.name = name
        self.directory = Path(directory)
        self.embedding_model = embedding_model
        self.embedding_cache = embedding_cache
        self.log = log

        self.openai = OpenAI(api_key=api_key)
        self.chroma = chromadb.PersistentClient(path=str(self.directory))
        # Метрика cosine: расстояние между векторами равно "1 - схожесть"
        self.collection = self.chroma.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def read_chunks(self, data_dir: Path, rules: Dict[str, Any], uniform: bool, clean: bool) -> List[Dict[str, Any]]:
        """
        Чтение документов папки и нарезка их на чанки.

        Args:
            data_dir: папка с документами
            rules: правила нарезки из chunking.txt
            uniform: одно правило на все файлы
            clean: очищать ли текст перед нарезкой

        Returns:
            чанки: text и metadata (source, file_type, strategy, chunk_index)
        """
        chunks = []

        for path in list_documents(data_dir):
            rule = find_rule(rules, path.name, uniform)
            # У файла нет строки в chunking.txt (или она закрыта решёткой): в базу не идёт
            if not rule["use"]:
                self.log(f"  {path.name}: нет в chunking.txt, пропущен")
                continue

            try:
                document = load_document(path, clean=clean)
                pieces = chunk_document(document, rule, uniform)
            except Exception as error:
                # Имя файла в ошибке: иначе по сообщению не понять, какой документ не прочитался
                raise ValueError(f"Не удалось обработать файл {path.name}: {error}") from error

            for number, piece in enumerate(pieces):
                chunks.append({
                    "text": piece["text"],
                    "metadata": {
                        "source": document.name,
                        "file_type": document.file_type,
                        "strategy": piece["strategy"],
                        "chunk_index": number
                    }
                })
            self.log(f"  {document.name}: {len(pieces)} чанков, способ {rule['strategy']}")

        return chunks

    def sync(self, data_dir: Path, rules: Dict[str, Any], uniform: bool, clean: bool) -> Dict[str, Any]:
        """
        Приведение индекса в соответствие с папкой документов и правилами нарезки.

        Документы читаются и режутся при каждом вызове: это быстро и не требует сети.
        По чанкам считается отпечаток. Если он совпал с сохранённым, база не трогается.
        Если нет, база собирается заново, а векторы для неизменившихся чанков берутся из кеша.

        Args:
            data_dir: папка с документами
            rules: правила нарезки из chunking.txt
            uniform: одно правило на все файлы
            clean: очищать ли текст перед нарезкой

        Returns:
            сведения о сборке: fingerprint (отпечаток, входит в ключ кеша ответов), chunks,
            read_seconds, rebuilt, а при пересборке - from_cache и computed (число векторов)
        """
        self.log(f"Чтение документов из {data_dir}...")
        started = time.perf_counter()
        chunks = self.read_chunks(data_dir, rules, uniform, clean)
        read_seconds = time.perf_counter() - started
        if not chunks:
            raise ValueError(f"В базу не попал ни один документ: проверьте папку {data_dir} и строки в chunking.txt")
        self.log(f"Чтение и нарезка: {len(chunks)} чанков за {read_seconds:.2f} с")

        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(repr((chunk["text"], sorted(chunk["metadata"].items()))).encode("utf-8"))
        info = {
            "fingerprint": digest.hexdigest(), "chunks": len(chunks), "read_seconds": read_seconds,
            "rebuilt": False, "from_cache": None, "computed": None
        }

        fingerprint_path = self.directory / f"{self.name}.fingerprint"
        saved = fingerprint_path.read_text(encoding="utf-8").strip() if fingerprint_path.exists() else None
        is_filled = self.collection.count() > 0
        if is_filled and saved == info["fingerprint"]:
            self.log("Документы и правила не менялись: база не пересобирается")
            return info

        texts = [chunk["text"] for chunk in chunks]
        # Векторы считаются до замены базы: при сбое OpenAI прежняя база остаётся целой
        vectors, info["from_cache"], info["computed"] = self._vectors_for(texts)

        if is_filled:
            self.log(f"Документы или правила нарезки изменились: коллекция '{self.name}' собирается заново")
            self.chroma.delete_collection(name=self.name)
            self.collection = self.chroma.create_collection(name=self.name, metadata={"hnsw:space": "cosine"})

        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start:start + ADD_BATCH_SIZE]
            self.collection.add(
                ids=[f"chunk_{number}" for number in range(start, start + len(batch))],
                documents=[chunk["text"] for chunk in batch],
                embeddings=vectors[start:start + ADD_BATCH_SIZE],
                metadatas=[chunk["metadata"] for chunk in batch]
            )

        fingerprint_path.write_text(info["fingerprint"], encoding="utf-8")
        info["rebuilt"] = True
        self.log(f"В коллекцию '{self.name}' загружено чанков: {len(chunks)}")
        return info

    def _vectors_for(self, texts: List[str]):
        """
        Векторы для чанков: готовые берутся из кеша эмбеддингов,
        в OpenAI уходят только тексты, которых в кеше ещё нет.

        Args:
            texts: тексты чанков

        Returns:
            векторы в порядке текстов; сколько текстов взято из кеша; сколько посчитано заново
        """
        known = self.embedding_cache.get_many(self.embedding_model, texts)
        # Одинаковые тексты (повторяющиеся строки таблицы) считаются один раз
        missing = list(dict.fromkeys(text for text in texts if text not in known))
        from_cache = sum(1 for text in texts if text in known)
        self.log(f"Векторы: из кеша {from_cache}, посчитать заново {len(missing)}")

        if missing:
            new_vectors = self._request_vectors(missing)
            self.embedding_cache.set_many(self.embedding_model, missing, new_vectors)
            known.update(zip(missing, new_vectors))

        return [known[text] for text in texts], from_cache, len(missing)

    def _request_vectors(self, texts: List[str]) -> List[List[float]]:
        """
        Запрос эмбеддингов у OpenAI, порциями.

        Args:
            texts: тексты

        Returns:
            векторы в порядке текстов
        """
        vectors = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            response = self.openai.embeddings.create(
                input=texts[start:start + EMBEDDING_BATCH_SIZE], model=self.embedding_model
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def search(self, question: str, limit: int) -> List[Dict[str, Any]]:
        """
        Поиск чанков, ближайших к вопросу по смыслу.

        Args:
            question: текст вопроса
            limit: сколько чанков вернуть

        Returns:
            чанки от самого похожего к наименее похожему: id, text, similarity, source, strategy
        """
        vector = self._request_vectors([question])[0]

        try:
            found = self._nearest(vector, limit)
        except NotFoundError:
            # Коллекцию пересобрал другой запуск программы (например, оценка в соседнем окне):
            # ссылка на прежнюю устарела, берём коллекцию заново по имени
            self.collection = self.chroma.get_collection(name=self.name)
            found = self._nearest(vector, limit)

        return [
            {
                "id": chunk_id,
                "text": text,
                "similarity": 1 - distance,
                "source": (metadata or {}).get("source", ""),
                "strategy": (metadata or {}).get("strategy", "")
            }
            for chunk_id, text, distance, metadata in zip(
                found["ids"][0], found["documents"][0], found["distances"][0], found["metadatas"][0]
            )
        ]

    def _nearest(self, vector: List[float], limit: int) -> Dict[str, Any]:
        """Запрос к ChromaDB; чанков запрашивается не больше, чем есть в коллекции."""
        return self.collection.query(query_embeddings=[vector], n_results=min(limit, self.collection.count()))

    def stats(self) -> Dict[str, Any]:
        """
        Сведения об индексе.

        Returns:
            имя коллекции, папка, число чанков всего и по файлам
        """
        metadatas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return {
            "name": self.name,
            "directory": str(self.directory),
            "chunks": self.collection.count(),
            "sources": dict(Counter(metadata.get("source", "") for metadata in metadatas))
        }
