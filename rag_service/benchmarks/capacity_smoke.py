"""Isolated capacity smoke. Does not touch a production index.

Example:
  python benchmarks/capacity_smoke.py --pages 40
  python benchmarks/capacity_smoke.py --pages 10000 --embedder minilm
Full 10000-page runs are intentionally not the default.
"""

import argparse
import tempfile
import time
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromadb.utils import embedding_functions

from adaptive.config import load_settings
from adaptive.contracts import AuthorizationScope, PageDocument
from adaptive.embeddings import HashEmbeddingFunction
from adaptive.engine import AdaptiveEngine
from adaptive.indexer import Indexer
from adaptive.store import StateStore
from adaptive.vector_index import VectorIndex


def run(pages: int, embedder: str = "hash-bow") -> dict:
    if embedder == "minilm":
        embedding = embedding_functions.DefaultEmbeddingFunction()
        model_id = "all-MiniLM-L6-v2"
    elif embedder == "hash-bow":
        embedding = HashEmbeddingFunction()
        model_id = "hash-bow"
    else:
        raise ValueError(f"Unsupported embedder: {embedder}")
    root = Path(tempfile.mkdtemp(prefix="rag-capacity-"))
    settings = replace(
        load_settings(),
        state_dir=str(root),
        chroma_dir=str(root / "chroma"),
        answer_mode="extractive",
        embedding_model_id=model_id,
    )
    store = StateStore(str(root / "rag_state.sqlite"))
    vectors = VectorIndex(str(root / "chroma"), "capacity_isolated", embedding)
    indexer = Indexer(store, vectors, settings)
    engine = AdaptiveEngine(store, vectors, settings)
    book_mod = 1000 if pages >= 1000 else 25
    shelf_mod = 50 if pages >= 1000 else 7
    started = time.perf_counter()
    for page_id in range(1, pages + 1):
        indexer.upsert(
            PageDocument(
                page_id=page_id,
                name=f"Sayfa {page_id}",
                markdown=f"Benzersiz kayıt {page_id} konusu KAPSAM-{page_id:05d} ve işlem adımı {page_id}.",
                book_name=f"Kitap {page_id % book_mod}",
                shelf_names=[f"Raf {page_id % shelf_mod}"],
                url=f"http://wiki.example/link/{page_id}",
                generation=1,
            )
        )
        if pages >= 500 and page_id % 500 == 0:
            print(f"indexed {page_id}", flush=True)
    index_s = time.perf_counter() - started
    scope = AuthorizationScope(
        principal="1",
        is_admin=True,
        can_use_ai=True,
        allowed_page_ids=None,
        fingerprint="admin",
        acl_version="all",
        issued_at=1,
        expires_at=10**12,
    )
    durations = []
    hits = []
    for page_id in (1, max(1, pages // 2), pages):
        begin = time.perf_counter()
        result = engine.answer(f"KAPSAM-{page_id:05d} işlemi nedir?", scope, diagnostics=True)
        durations.append(time.perf_counter() - begin)
        hits.append(page_id in (result.get("evidence_page_ids") or []))
    durations.sort()
    if embedder == "minilm" and pages >= 10000:
        note = "Synthetic one-sentence pages with all-MiniLM-L6-v2 in a temp directory. Not a 100000-chunk corpus."
    else:
        note = "Not a production MiniLM or 10000-page capacity result."
    return {
        "pages": pages,
        "chunks": vectors.collection.count(),
        "index_seconds": round(index_s, 3),
        "search_p50_seconds": round(durations[len(durations) // 2], 4),
        "expected_page_hits": hits,
        "embedder": model_id if embedder == "minilm" else "hash-bow",
        "note": note,
        "directory": str(root),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=40)
    parser.add_argument("--embedder", choices=("hash-bow", "minilm"), default="hash-bow")
    args = parser.parse_args()
    print(run(args.pages, args.embedder))
