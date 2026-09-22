import hashlib
import math
import os
import re
from typing import Dict, List

import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

from adaptive.provider import redact_secrets


_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words vectors for offline tests.

    This is not all-MiniLM-L6-v2 and must not be reported as semantic quality.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text or "") for text in input]

    @staticmethod
    def name() -> str:
        return "hash-bow"

    def get_config(self) -> Dict[str, int]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, int]) -> "HashEmbeddingFunction":
        return HashEmbeddingFunction(dim=int(config.get("dim", 64)))

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def max_tokens(self) -> int:
        return 256

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Gemini embedding model for indexing and query vectors.

    The API key is sent in a header. Vectors are L2-normalized so a reduced
    output dimension remains usable for cosine search.
    """

    def __init__(
        self,
        model_id: str = "gemini-embedding-001",
        api_key: str = "",
        output_dimensionality: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT",
        timeout_s: float = 60.0,
        batch_size: int = 32,
        transport: object = None,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        self.transport = transport

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed(list(input), self.task_type)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    @staticmethod
    def name() -> str:
        return "gemini-embedding-001"

    def get_config(self) -> Dict[str, str]:
        return {"model_id": self.model_id, "output_dimensionality": str(self.output_dimensionality)}

    @staticmethod
    def build_from_config(config: Dict[str, str]) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(
            model_id=str(config.get("model_id") or "gemini-embedding-001"),
            output_dimensionality=int(config.get("output_dimensionality") or 768),
        )

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def max_tokens(self) -> int:
        return 2048

    def _embed(self, texts: List[str], task_type: str) -> Embeddings:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
        vectors: Embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text if text else " " for text in texts[start : start + self.batch_size]]
            vectors.extend(self._embed_batch(batch, task_type))
        return vectors

    def _embed_batch(self, texts: List[str], task_type: str) -> Embeddings:
        model = self.model_id
        if not model.startswith("models/"):
            model = f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": model,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.output_dimensionality,
                }
                for text in texts
            ]
        }
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                response = client.post(
                    url,
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            cleaned = redact_secrets(str(exc), [self.api_key])
            raise RuntimeError(f"Gemini embedding failed: {cleaned}") from None
        rows = data.get("embeddings") or []
        if len(rows) != len(texts):
            raise RuntimeError("Gemini embedding failed: incomplete batch")
        return [_normalize(list(row.get("values") or [])) for row in rows]


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def build_embedding(model_id: str = ""):
    chosen = model_id or os.getenv("EMBEDDING_MODEL_ID", "gemini-embedding-001")
    if chosen == "hash-bow":
        return HashEmbeddingFunction()
    if chosen == "all-MiniLM-L6-v2":
        return embedding_functions.DefaultEmbeddingFunction()
    return GeminiEmbeddingFunction(model_id=chosen)
