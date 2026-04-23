from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import json
from openai import OpenAI
import os

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class MemoryItem:
    memory_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class InMemoryVectorStore:
    """A minimal thread-safe in-memory vector store for agent memories.

    This is intentionally simple and dependency-free. If you need persistence or
    ANN, swap this with a proper store and keep the same interface.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: Dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.memory_id] = item

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock:
            return self._items.get(memory_id)

    def update(self, memory_id: str, new_content: Optional[str] = None, new_metadata: Optional[Dict[str, str]] = None) -> bool:
        with self._lock:
            item = self._items.get(memory_id)
            if item is None:
                return False
            if new_content is not None:
                item.content = new_content
            if new_metadata is not None:
                item.metadata.update(new_metadata)
            return True

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            return self._items.pop(memory_id, None) is not None

    def clear(self) -> None:
        """Remove all stored memories."""
        with self._lock:
            self._items.clear()

    def search(self, query_embedding: List[float], top_k: int = 5, metadata_filter: Optional[Dict[str, str]] = None) -> List[Tuple[MemoryItem, float]]:
        with self._lock:
            scored: List[Tuple[MemoryItem, float]] = []
            for item in self._items.values():
                if metadata_filter and not all(item.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
                if item.embedding is None:
                    continue
                score = _cosine_similarity(query_embedding, item.embedding)
                if score > 0.0:
                    scored.append((item, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[: max(1, top_k)]


class MemoryManager:
    """High-level memory manager wrapping the vector store and an embedding fn."""

    def __init__(self, embedding_model, embedding_dim) -> None:
        self._store = InMemoryVectorStore()
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable is not set. "
                "Please set it before running the workflow, e.g., export DASHSCOPE_API_KEY='your_key'"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    def embed(self, content: str):
        completion = self.client.embeddings.create(
        model=self.embedding_model,
        input=content,
        dimensions=self.embedding_dim,
        encoding_format="float"
        )
        json_response = completion.model_dump_json()
        response = json.loads(json_response)
        return response['data'][0]['embedding']
        

    def add_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, str]] = None) -> None:
        if not content:
            return
        embedding = self.embed(content)
        self._store.add(MemoryItem(memory_id=memory_id, content=content, metadata=metadata or {}, embedding=embedding))

    def update_memory(self, memory_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> bool:
        # If content is updated, refresh embedding
        if content is not None:
            embedding = self.embed(content)
            item = self._store.get(memory_id)
            if item is None:
                return False
            item.embedding = embedding
        return self._store.update(memory_id, content, metadata)

    def delete_memory(self, memory_id: str) -> bool:
        return self._store.delete(memory_id)

    def clear(self) -> None:
        """Remove all memories from the store."""
        self._store.clear()

    def retrieve(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, str]] = None) -> List[MemoryItem]:
        if not query:
            return []
        q_emb = self.embed(query)
        return [it for it, _ in self._store.search(q_emb, top_k=top_k, metadata_filter=metadata_filter)]

class chat_client():
    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable is not set. "
                "Please set it before running the workflow, e.g., export DASHSCOPE_API_KEY='your_key'"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def chat(self, messages: List[Dict], model_name: str = "qwen-max") -> str:
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return completion.choices[0].message.content
