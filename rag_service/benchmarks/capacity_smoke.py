"""Isolated capacity smoke. Does not touch a production index.

Example:
  python benchmarks/capacity_smoke.py --pages 40
  python benchmarks/capacity_smoke.py --pages 10000 --sections 12 --words-per-section 80 --embedder minilm
Full 10000-page runs are intentionally not the default.
This script never calls Gemini or an answer model.
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


def run(pages: int, embedder: str = "hash-bow", sections: int = 1, words_per_section: int = 10, queries: int = 20) -> dict:
    if pages < 1 or sections < 1 or words_per_section < 1 or queries < 1:
        raise ValueError("pages, sections, words_per_section, and queries must be positive")
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
        filler = " ".join((["process", "owner", "review", "checklist"] * ((words_per_section + 3) // 4))[:words_per_section])
        markdown = "\n\n".join(
            f"## Section {section}\n\nKAPSAM-{page_id:05d}-{section:02d} {filler}"
            for section in range(1, sections + 1)
        )
        indexer.upsert(
            PageDocument(
                page_id=page_id,
                name=f"Sayfa {page_id}",
                markdown=markdown,
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
    sampled_ids = sorted({1, pages, *(1 + (pages - 1) * index // max(1, queries - 1) for index in range(queries))})
    for page_id in sampled_ids:
        begin = time.perf_counter()
        result = engine.answer(f"KAPSAM-{page_id:05d}-01 process", scope, diagnostics=True)
        durations.append(time.perf_counter() - begin)
        hits.append(page_id in (result.get("evidence_page_ids") or []))
    durations.sort()
    note = "Isolated synthetic corpus and sequential queries; no Gemini, real BookStack content, or concurrent load."
    return {
        "pages": pages,
        "sections_per_page": sections,
        "words_per_section": words_per_section,
        "chunks": vectors.collection.count(),
        "chunks_per_page": round(vectors.collection.count() / pages, 2),
        "index_seconds": round(index_s, 3),
        "search_p50_seconds": round(durations[len(durations) // 2], 4),
        "search_p95_seconds": round(durations[min(len(durations) - 1, max(0, (95 * len(durations) + 99) // 100 - 1))], 4),
        "queries": len(durations),
        "expected_page_hits": hits,
        "embedder": model_id if embedder == "minilm" else "hash-bow",
        "note": note,
        "directory": str(root),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=40)
    parser.add_argument("--embedder", choices=("hash-bow", "minilm"), default="hash-bow")
    parser.add_argument("--sections", type=int, default=1)
    parser.add_argument("--words-per-section", type=int, default=10)
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    print(run(args.pages, args.embedder, args.sections, args.words_per_section, args.queries))
