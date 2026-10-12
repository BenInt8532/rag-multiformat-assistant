"""
Ассистент по документам.

Путь вопроса: кеш ответов -> поиск чанков в индексе -> отбор контекста -> модель -> кеш ответов.
"""

import hashlib
import os
import time
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from answer_cache import AnswerCache
from config import ANSWER_CACHE_DB_PATH, CHROMA_DIR, DATA_DIR, EMBEDDING_CACHE_DB_PATH, settings_key
from embedding_cache import EmbeddingCache
from index import ChunkIndex, collection_name_for
from prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, USER_PROMPT, build_messages, split_sources

# Настройки, от которых зависит ответ на вопрос: входят в ключ кеша ответов
ANSWER_SETTINGS = ("model", "temperature", "top_k", "context_budget", "score_threshold", "embedding_model")
# Наибольшая длина ответа модели в токенах
MAX_ANSWER_TOKENS = 500


class Assistant:
    """Ассистент, который отвечает на вопросы по базе документов."""

    def __init__(self, settings: Dict[str, Any], rules: Dict[str, Any], log: Callable[[str], None] = print):
        """
        Args:
            settings: настройки из config.get_settings
            rules: правила нарезки из config.load_rules
            log: куда писать ход работы; чтобы ассистент молчал, передайте функцию, которая ничего не делает
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не задан: создайте файл .env по образцу env.example")

        self.settings = settings
        self.log = log
        self.openai = OpenAI(api_key=api_key)

        self.index = ChunkIndex(
            name=collection_name_for(settings),
            directory=CHROMA_DIR,
            embedding_model=settings["embedding_model"],
            api_key=api_key,
            embedding_cache=EmbeddingCache(str(EMBEDDING_CACHE_DB_PATH)),
            log=log
        )
        # Сведения о сборке базы: сколько чанков, пересобиралась ли, сколько векторов из кеша
        self.build_info = self.index.sync(
            DATA_DIR, rules,
            uniform=settings["chunking_mode"] == "uniform",
            clean=settings["clean_text"]
        )

        # Область кеша ответов: настройки, текст промпта и отпечаток базы. Ответ, полученный при других
        # настройках, с другим промптом или по другим документам, в кеше не находится
        prompt_digest = hashlib.sha256(
            f"{SYSTEM_PROMPT}|{USER_PROMPT}|{MAX_ANSWER_TOKENS}".encode("utf-8")
        ).hexdigest()[:16]
        self.answers = AnswerCache(
            str(ANSWER_CACHE_DB_PATH),
            scope=(f"{settings_key(settings, ANSWER_SETTINGS)}|prompt={prompt_digest}"
                   f"|docs={self.build_info['fingerprint']}")
        )

    def select_context(self, found: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Отбор чанков для контекста из найденных поиском.

        Чанки слабее порога схожести отбрасываются. Если задан бюджет контекста,
        лучшие чанки набираются, пока помещаются в бюджет: мелких войдёт больше, крупных меньше.

        Args:
            found: чанки от поиска, от самого похожего к наименее похожему

        Returns:
            чанки, которые пойдут в промпт
        """
        threshold = self.settings["score_threshold"]
        if threshold is not None:
            found = [chunk for chunk in found if chunk["similarity"] >= threshold]

        budget = self.settings["context_budget"]
        if budget is None:
            return found

        selected = []
        used = 0
        for chunk in found:
            # Первый чанк берётся всегда, даже если он один больше бюджета
            if selected and used + len(chunk["text"]) > budget:
                break
            selected.append(chunk)
            used += len(chunk["text"])
        return selected

    def ask(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Ответ на вопрос.

        Args:
            question: вопрос пользователя
            use_cache: брать ли готовый ответ из кеша и сохранять ли новый

        Returns:
            словарь: question, answer (без строки источников), cited (файлы, на которые сослалась модель),
            from_cache, context (чанки в промпте), found (все найденные чанки),
            model_called, usage (токены запроса и ответа), timing (секунды: search, generation, total)
        """
        started = time.perf_counter()

        if use_cache:
            cached = self.answers.get(question)
            if cached:
                self.log("Ответ взят из кеша")
                # В кеше лежит ответ модели как есть: строка источников отделяется при чтении
                answer, cited = split_sources(cached["answer"], cached["sources"])
                return {
                    "question": question,
                    "answer": answer,
                    "cited": cited,
                    "from_cache": True,
                    "saved_at": cached["saved_at"],
                    "context": cached["sources"],
                    "found": [],
                    "model_called": False,
                    "usage": None,
                    "timing": {"search": 0.0, "generation": 0.0, "total": time.perf_counter() - started}
                }

        search_started = time.perf_counter()
        found = self.index.search(question, limit=self.settings["top_k"])
        search_seconds = time.perf_counter() - search_started

        context = self.select_context(found)
        self.log(f"Найдено чанков: {len(found)}, в контекст отобрано: {len(context)}")

        usage: Optional[Dict[str, int]] = None
        generation_seconds = 0.0
        if context:
            generation_started = time.perf_counter()
            answer, usage = self._generate(question, context)
            generation_seconds = time.perf_counter() - generation_started
        else:
            self.log("Ни один чанк не прошёл отбор: модель не вызывается")
            answer = NO_CONTEXT_ANSWER

        if use_cache:
            self.answers.put(
                question, answer,
                [{"source": chunk["source"], "text": chunk["text"]} for chunk in context]
            )

        answer, cited = split_sources(answer, context)

        return {
            "question": question,
            "answer": answer,
            "cited": cited,
            "from_cache": False,
            "context": context,
            "found": found,
            "model_called": bool(context),
            "usage": usage,
            "timing": {
                "search": search_seconds,
                "generation": generation_seconds,
                "total": time.perf_counter() - started
            }
        }

    def _generate(self, question: str, context: List[Dict[str, Any]]):
        """
        Запрос ответа у модели.

        Args:
            question: вопрос пользователя
            context: чанки для промпта

        Returns:
            текст ответа и число токенов запроса и ответа
        """
        response = self.openai.chat.completions.create(
            model=self.settings["model"],
            messages=build_messages(question, context),
            temperature=self.settings["temperature"],
            max_tokens=MAX_ANSWER_TOKENS
        )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        return (response.choices[0].message.content or "").strip(), usage

    def stats(self) -> Dict[str, Any]:
        """Сведения об индексе, кеше ответов, кеше эмбеддингов и настройках."""
        return {
            "index": self.index.stats(),
            "answers": self.answers.stats(),
            "embeddings": self.index.embedding_cache.count(),
            "settings": self.settings
        }
