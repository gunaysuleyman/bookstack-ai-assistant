import os
from typing import List, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings


def _model_id(embedding_function) -> str:
    name = getattr(embedding_function, "name", None)
    if callable(name):
        try:
            return str(name())
        except TypeError:
            return str(name)
    return str(name or "")


class VectorIndex:
    def __init__(self, path: str, collection_name: str, embedding_function):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.embedding_model_id = _model_id(embedding_function)
        self.embed_calls = 0
        self.client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._open_collection()

    def _open_collection(self):
        metadata = {"hnsw:space": "cosine", "embedding_model": self.embedding_model_id}
        try:
            existing = self.client.get_collection(name=self.collection_name)
        except Exception:
            existing = None
        if existing is not None and (existing.metadata or {}).get("embedding_model") != self.embedding_model_id:
            self.client.delete_collection(self.collection_name)
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=metadata,
        )

    def add(self, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[dict]) -> None:
        if not ids:
            return
        self.embed_calls += 1
        self.collection.upsert(ids=list(ids), documents=list(documents), metadatas=list(metadatas))

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=list(ids))

    def query(self, text: str, n_results: int, where: Optional[dict] = None) -> List[dict]:
        if n_results <= 0:
            return []
        try:
            available = self.collection.count()
        except Exception:
            available = n_results
        if available <= 0:
            return []
        n_results = min(n_results, available)
        kwargs = {
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if "embed_query" in type(self.embedding_function).__dict__:
            kwargs["query_embeddings"] = [self.embedding_function.embed_query(text)]
        else:
            kwargs["query_texts"] = [text]
        try:
            result = self.collection.query(**kwargs)
        except Exception:
            return []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        rows = []
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document": doc,
                    "metadata": meta or {},
                    "distance": distance,
                }
            )
        return rows
