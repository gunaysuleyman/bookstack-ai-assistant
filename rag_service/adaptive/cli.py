import argparse
import json

from adaptive.config import load_settings, manifest
from adaptive.embeddings import build_embedding
from adaptive.indexer import Indexer
from adaptive.store import StateStore
from adaptive.vector_index import VectorIndex


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BookStack RAG index operations")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("manifest")
    sub.add_parser("recover")
    retry = sub.add_parser("retry-dead")
    retry.add_argument("job_id", type=int)
    args = parser.parse_args(argv)
    settings = load_settings()
    store = StateStore(f"{settings.state_dir}/rag_state.sqlite")
    try:
        if args.command == "status":
            print(json.dumps(store.job_counts(), ensure_ascii=False))
            return 0
        if args.command == "manifest":
            print(json.dumps(manifest(settings), ensure_ascii=False, indent=2))
            return 0
        if args.command == "retry-dead":
            ok = store.retry_dead(args.job_id)
            print(json.dumps({"retried": ok, "job_id": args.job_id}))
            return 0 if ok else 1
        embedding = build_embedding(settings.embedding_model_id)
        vectors = VectorIndex(settings.chroma_dir, settings.adaptive_collection, embedding)
        indexer = Indexer(store, vectors, settings)
        actions = indexer.recover()
        print(json.dumps({"actions": actions}, ensure_ascii=False))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
