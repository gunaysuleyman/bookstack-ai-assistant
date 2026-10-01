"""Isolated scale and hierarchy stress test. Does not touch a production index.

It builds a synthetic BookStack library (shelves -> books -> chapters -> pages),
indexes it through the real Adaptive `Indexer`, and queries it through the real
`document_search` tool path (`ToolRegistry` + `HybridSearcher`).

Part A, scale: indexing throughput per window, retrieval recall and latency at
checkpoints, restricted ACL scopes, concurrent queries, catalog correctness,
and incremental updates on a full index.

Part B, hierarchy: whether shelf, book, chapter and page-title names reach
retrieval and catalog tools. These names never appear in page bodies, so a hit
can only come from hierarchy metadata.

Examples:
  python benchmarks/stress_scale.py --preset quick
  python benchmarks/stress_scale.py --preset full --report ../docs/stress_results.md

This script never calls Gemini or an answer model. The default hash-bow
embedder is not a semantic model: recall numbers measure index mechanics at
scale, not semantic quality.
"""

import argparse
import json
import random
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive.config import load_settings
from adaptive.contracts import AuthorizationScope, PageDocument
from adaptive.embeddings import HashEmbeddingFunction
from adaptive.hybrid import HybridSearcher, fts_match
from adaptive.indexer import Indexer
from adaptive.store import StateStore
from adaptive.tools import FUNCTION_DECLARATIONS, ToolRegistry
from adaptive.vector_index import VectorIndex


PRESETS = {
    "quick": {"books": 50, "pages": 500, "shelves": 8, "checkpoints": [250, 500]},
    "medium": {"books": 200, "pages": 2000, "shelves": 20, "checkpoints": [500, 1000, 2000]},
    "full": {"books": 1000, "pages": 10000, "shelves": 60, "checkpoints": [1000, 2500, 5000, 10000]},
}

DEPARTMENTS = [
    "Finans", "İnsan Kaynakları", "Bilgi Teknolojileri", "Satın Alma", "Hukuk", "Pazarlama",
    "Lojistik", "Üretim", "Kalite", "Müşteri Hizmetleri", "Ar-Ge", "Güvenlik",
]
REGIONS = ["Ankara", "İzmir", "Bursa", "Adana", "Konya", "Trabzon", "Eskişehir", "Samsun", "Kayseri", "Antalya"]
CHAPTER_THEMES = [
    "Tedarik Süreçleri", "Bordro İşlemleri", "Ağ Altyapısı", "Sözleşme Yönetimi", "Kampanya Planlama",
    "Depo Operasyonları", "Bakım Planı", "Denetim Hazırlığı", "Şikayet Yönetimi", "Prototip Testleri",
    "Erişim Kontrolü", "Bütçe Kapanışı", "İşe Alım", "Yedekleme Politikası", "Olay Müdahalesi",
    "Tedarikçi Değerlendirme", "Eğitim Programı", "Envanter Sayımı", "Sevkiyat Kuralları", "Arşivleme",
]
COLORS = [
    "Mavi", "Kırmızı", "Yeşil", "Turuncu", "Mor", "Gri", "Lacivert", "Bordo", "Fıstık", "Turkuaz",
    "Kehribar", "Zümrüt", "Safir", "Bakır", "Gümüş", "Altın", "Kömür", "Leylak", "Mercan", "Zeytin",
]
ANIMALS = [
    "Pelikan", "Vaşak", "Albatros", "Kunduz", "Martı", "Sincap", "Ceylan", "Balaban", "Yunus", "Porsuk",
    "Atmaca", "Kartal", "Tilki", "Kaplan", "Baykuş", "Flamingo", "Leylek", "Ayı", "Geyik", "Şahin",
    "Turna", "Balina", "Kurt", "Keçi", "Serçe", "Karga", "Doğan", "Ördek", "Bizon", "Zebra",
    "Panda", "Koala", "Lama", "Fok", "Penguen", "Kanguru", "Ahtapot", "Mersin", "Levrek", "Kirpi",
    "Sansar", "Gelincik", "Bukalemun", "Iguana", "Kobra", "Piton", "Tavus", "Papağan", "Kumru", "Saka",
]
TOPIC_TITLES = [
    "Kurulum Notları", "Çalışma Talimatı", "Kontrol Listesi", "Süreç Akışı", "Sık Sorulanlar",
    "Rol ve Sorumluluklar", "Onay Adımları", "Raporlama Kuralları", "Yapılandırma Rehberi", "Geçiş Planı",
]
FIRST = ["Ayşe", "Mehmet", "Zeynep", "Ali", "Elif", "Can", "Deniz", "Burak", "Selin", "Emre", "Derya", "Kerem"]
LAST = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Aydın", "Öztürk", "Arslan", "Doğan", "Koç", "Kurt", "Aksoy"]
FILLER = (
    "süreç sahibi talebi kayıt altına alır ve ilgili ekip onay adımlarını takip eder "
    "form eksiksiz doldurulmadan işlem başlatılmaz ve değişiklikler haftalık toplantıda gözden geçirilir "
    "sistem kayıtları düzenli aralıklarla kontrol edilir ve sapmalar raporlanır "
    "yetkisiz erişim girişimleri güvenlik ekibine bildirilir ve kayıt numarası ile izlenir "
    "belgeler güncel sürüm numarası ile arşivlenir ve eski sürümler salt okunur tutulur"
).split()
AMBIGUOUS_FACTS = [
    ("seyahat avansı onay limiti", "Seyahat avansı onay limiti {v} TL olarak uygulanır."),
    ("fazla mesai bildirim süresi", "Fazla mesai bildirim süresi {v} gündür."),
    ("parola değiştirme periyodu", "Parola değiştirme periyodu {v} gün olarak belirlenmiştir."),
    ("fatura itiraz süresi", "Fatura itiraz süresi {v} iş günüdür."),
    ("demirbaş sayım sıklığı", "Demirbaş sayım sıklığı yılda {v} kezdir."),
    ("yedek saklama süresi", "Yedek saklama süresi {v} gündür."),
    ("numune bekletme süresi", "Numune bekletme süresi {v} saattir."),
    ("kampanya bütçe tavanı", "Kampanya bütçe tavanı {v} bin TL'dir."),
]


@dataclass
class Shelf:
    shelf_id: int
    name: str
    book_ids: List[int] = field(default_factory=list)


@dataclass
class Chapter:
    chapter_id: int
    book_id: int
    name: str
    page_ids: List[int] = field(default_factory=list)


@dataclass
class Book:
    book_id: int
    name: str
    shelf_names: List[str]
    chapter_ids: List[int] = field(default_factory=list)
    page_ids: List[int] = field(default_factory=list)


@dataclass
class Library:
    shelves: Dict[int, Shelf]
    books: Dict[int, Book]
    chapters: Dict[int, Chapter]
    pages: Dict[int, PageDocument]
    needles: Dict[int, dict]
    ambiguous: List[dict]


def _code(rng: random.Random, used: set) -> str:
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    while True:
        code = "".join(rng.choice(alphabet) for _ in range(6))
        if code not in used and any(ch.isdigit() for ch in code) and any(ch.isalpha() for ch in code):
            used.add(code)
            return code


def _filler(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(FILLER) for _ in range(words)).capitalize() + "."


def build_library(books: int, pages: int, shelves: int, sections: int, words: int, seed: int = 7) -> Library:
    rng = random.Random(seed)
    shelf_map: Dict[int, Shelf] = {}
    for shelf_id in range(1, shelves + 1):
        dept = DEPARTMENTS[(shelf_id - 1) % len(DEPARTMENTS)]
        region = REGIONS[((shelf_id - 1) // len(DEPARTMENTS)) % len(REGIONS)]
        suffix = "" if shelf_id <= len(DEPARTMENTS) * len(REGIONS) else f" {shelf_id}"
        shelf_map[shelf_id] = Shelf(shelf_id, f"{dept} {region} Rafı{suffix}")
    book_map: Dict[int, Book] = {}
    chapter_map: Dict[int, Chapter] = {}
    chapter_id = 0
    for book_id in range(1, books + 1):
        roll = rng.random()
        if roll < 0.03:
            on = []
        elif roll < 0.13:
            on = rng.sample(sorted(shelf_map), k=min(2, len(shelf_map)))
        else:
            on = [rng.choice(sorted(shelf_map))]
        for sid in on:
            shelf_map[sid].book_ids.append(book_id)
        book = Book(book_id, f"{COLORS[book_id % len(COLORS)]} Kılavuz {book_id:04d}", [shelf_map[s].name for s in on])
        for _ in range(rng.randint(0, 4)):
            chapter_id += 1
            theme = CHAPTER_THEMES[chapter_id % len(CHAPTER_THEMES)]
            chapter_map[chapter_id] = Chapter(chapter_id, book_id, f"{theme} {chapter_id:04d}")
            book.chapter_ids.append(chapter_id)
        book_map[book_id] = book

    used_codes: set = set()
    page_map: Dict[int, PageDocument] = {}
    needles: Dict[int, dict] = {}
    title_combos = [(c, a, t) for t in TOPIC_TITLES for a in ANIMALS for c in COLORS]
    rng.shuffle(title_combos)
    for page_id in range(1, pages + 1):
        book = book_map[(page_id - 1) % books + 1]
        chapter = None
        if book.chapter_ids and rng.random() < 0.8:
            chapter = chapter_map[rng.choice(book.chapter_ids)]
        color, animal, topic = title_combos[(page_id - 1) % len(title_combos)]
        title = f"{color} {animal} {topic}" if page_id <= len(title_combos) else f"{color} {animal} {topic} {page_id}"
        code = _code(rng, used_codes)
        person = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        ext = rng.randint(1000, 9999)
        blocks = []
        needle_section = rng.randrange(sections)
        for section in range(sections):
            heading = rng.choice(["Amaç", "Kapsam", "Uygulama", "Sorumluluklar", "Kontroller", "Kayıtlar", "İstisnalar"])
            body = _filler(rng, words)
            if section == needle_section:
                body += f" {code} sisteminin yedekleme sorumlusu {person} olup dahili hattı {ext} numarasıdır."
            blocks.append(f"## {heading}\n\n{body}")
        page_map[page_id] = PageDocument(
            page_id=page_id,
            name=title,
            markdown="\n\n".join(blocks),
            book_id=book.book_id,
            book_name=book.name,
            chapter_id=chapter.chapter_id if chapter else 0,
            chapter_name=chapter.name if chapter else "General Chapter",
            shelf_names=list(book.shelf_names),
            url=f"http://wiki.example/books/{book.book_id}/page/{page_id}",
            generation=1,
        )
        book.page_ids.append(page_id)
        if chapter:
            chapter.page_ids.append(page_id)
        needles[page_id] = {"code": code, "person": person, "ext": ext, "title": title}

    ambiguous = _plant_ambiguous_facts(rng, page_map, book_map, chapter_map)
    return Library(shelf_map, book_map, chapter_map, page_map, needles, ambiguous)


def _plant_ambiguous_facts(rng, page_map, book_map, chapter_map, per_fact: int = 4, groups_per_fact: int = 5) -> List[dict]:
    """Same sentence on K pages with different values; only hierarchy names can disambiguate."""
    eligible = [
        p for p in page_map.values()
        if p.chapter_id and len(p.shelf_names) == 1
    ]
    rng.shuffle(eligible)
    used_pages = set()
    groups = []
    cursor = 0
    for topic, template in AMBIGUOUS_FACTS:
        for _ in range(groups_per_fact):
            chosen: List[PageDocument] = []
            shelves_seen, books_seen, chapters_seen = set(), set(), set()
            while cursor < len(eligible) and len(chosen) < per_fact:
                page = eligible[cursor]
                cursor += 1
                shelf = page.shelf_names[0]
                if page.page_id in used_pages or shelf in shelves_seen or page.book_id in books_seen:
                    continue
                if page.chapter_name in chapters_seen:
                    continue
                chosen.append(page)
                shelves_seen.add(shelf)
                books_seen.add(page.book_id)
                chapters_seen.add(page.chapter_name)
            if len(chosen) < per_fact:
                return groups
            members = []
            for page in chosen:
                value = rng.randint(2, 950)
                sentence = template.format(v=value)
                page_map[page.page_id] = page.model_copy(
                    update={"markdown": page.markdown + f"\n\n## Parametreler\n\n{sentence}"}
                )
                used_pages.add(page.page_id)
                members.append(
                    {
                        "page_id": page.page_id,
                        "value": value,
                        "shelf": page.shelf_names[0],
                        "book": page.book_name,
                        "chapter": page.chapter_name,
                        "title": page.name,
                    }
                )
            groups.append({"topic": topic, "members": members})
    return groups


def admin_scope() -> AuthorizationScope:
    return AuthorizationScope(
        principal="1", is_admin=True, can_use_ai=True, allowed_page_ids=None,
        fingerprint="admin", acl_version="all", issued_at=1, expires_at=10**12,
    )


def user_scope(page_ids: Sequence[int]) -> AuthorizationScope:
    return AuthorizationScope(
        principal="2", is_admin=False, can_use_ai=True, allowed_page_ids=sorted(page_ids),
        fingerprint=f"user-{len(page_ids)}", acl_version="v1", issued_at=1, expires_at=10**12,
    )


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100 * len(ordered) + 0.5)) - 1))
    return ordered[index]


class Harness:
    def __init__(self, root: Path, embedder: str):
        if embedder == "minilm":
            from chromadb.utils import embedding_functions

            embedding = embedding_functions.DefaultEmbeddingFunction()
            model_id = "all-MiniLM-L6-v2"
        else:
            embedding = HashEmbeddingFunction()
            model_id = "hash-bow"
        self.settings = replace(
            load_settings(), state_dir=str(root), chroma_dir=str(root / "chroma"),
            answer_mode="extractive", embedding_model_id=model_id,
        )
        self.store = StateStore(str(root / "rag_state.sqlite"))
        self.vectors = VectorIndex(str(root / "chroma"), "stress_isolated", embedding)
        self.indexer = Indexer(self.store, self.vectors, self.settings)
        self.searcher = HybridSearcher(self.store, self.vectors, acl_batch=self.settings.acl_batch)
        self.tools = ToolRegistry(self.store, self.searcher, self.settings)

    def search(self, query: str, scope: AuthorizationScope, limit: int = 8, **containers) -> List[dict]:
        result = self.tools.execute("document_search", {"query": query, "limit": limit, **containers}, scope)
        return result.get("passages") or []


def _page_ids(passages: List[dict]) -> List[int]:
    ordered = []
    for passage in passages:
        pid = int(passage["page_id"])
        if pid not in ordered:
            ordered.append(pid)
    return ordered


def eval_needles(h: Harness, lib: Library, indexed: int, samples: int, rng: random.Random) -> dict:
    """Unique-code questions. Every page carries the same sentence template; only the code differs."""
    ids = rng.sample(range(1, indexed + 1), k=min(samples, indexed))
    scope = admin_scope()
    hit1 = hit8 = 0
    vector_hit = lexical_hit = 0
    durations = []
    for page_id in ids:
        needle = lib.needles[page_id]
        query = f"{needle['code']} sisteminin yedekleme sorumlusu kim?"
        began = time.perf_counter()
        passages = h.search(query, scope)
        durations.append(time.perf_counter() - began)
        pages = _page_ids(passages)
        hit1 += bool(pages[:1] == [page_id])
        hit8 += page_id in pages
        vector_rows = h.searcher._vector_ranks(query, None, 8)
        lexical_rows = h.searcher._lexical_ranks(query, None, h.store.active_revision_map(), 8)
        chunks = h.store.get_chunks(vector_rows + lexical_rows)
        vector_hit += any(int(chunks[c]["page_id"]) == page_id for c in vector_rows if c in chunks)
        lexical_hit += any(int(chunks[c]["page_id"]) == page_id for c in lexical_rows if c in chunks)
    n = len(ids)
    return {
        "queries": n,
        "recall_at_1": round(hit1 / n, 3),
        "recall_at_8": round(hit8 / n, 3),
        "vector_only_recall_at_8": round(vector_hit / n, 3),
        "lexical_only_recall_at_8": round(lexical_hit / n, 3),
        "latency_p50_ms": round(pct(durations, 50) * 1000, 1),
        "latency_p95_ms": round(pct(durations, 95) * 1000, 1),
        "latency_p99_ms": round(pct(durations, 99) * 1000, 1),
    }


def eval_acl(h: Harness, lib: Library, indexed: int, sizes: Sequence[int], samples: int, rng: random.Random) -> List[dict]:
    rows = []
    universe = list(range(1, indexed + 1))
    for size in sizes:
        if size >= indexed:
            continue
        allowed = set(rng.sample(universe, k=size))
        scope = user_scope(sorted(allowed))
        denied = [p for p in universe if p not in allowed]
        leaks = 0
        hits = 0
        durations = []
        probe_allowed = rng.sample(sorted(allowed), k=min(samples, len(allowed)))
        probe_denied = rng.sample(denied, k=min(samples, len(denied)))
        for page_id in probe_allowed:
            began = time.perf_counter()
            passages = h.search(f"{lib.needles[page_id]['code']} sisteminin yedekleme sorumlusu kim?", scope)
            durations.append(time.perf_counter() - began)
            pages = _page_ids(passages)
            hits += page_id in pages
            leaks += sum(1 for p in pages if p not in allowed)
        for page_id in probe_denied:
            passages = h.search(f"{lib.needles[page_id]['code']} sisteminin yedekleme sorumlusu kim?", scope)
            leaks += sum(1 for p in _page_ids(passages) if p not in allowed)
        rows.append(
            {
                "allowed_pages": size,
                "acl_batches": (size + h.settings.acl_batch - 1) // h.settings.acl_batch,
                "recall_at_8": round(hits / max(1, len(probe_allowed)), 3),
                "leaked_pages": leaks,
                "latency_p50_ms": round(pct(durations, 50) * 1000, 1),
                "latency_p95_ms": round(pct(durations, 95) * 1000, 1),
            }
        )
    return rows


def eval_concurrency(h: Harness, lib: Library, indexed: int, threads: int, per_thread: int, rng: random.Random) -> dict:
    scope = admin_scope()
    queries = [
        f"{lib.needles[p]['code']} sisteminin yedekleme sorumlusu kim?"
        for p in rng.sample(range(1, indexed + 1), k=min(indexed, threads * per_thread))
    ]
    durations: List[float] = []
    errors: List[str] = []
    lock = threading.Lock()

    def worker(chunk: List[str]) -> None:
        for query in chunk:
            began = time.perf_counter()
            try:
                h.search(query, scope)
            except Exception as exc:  # pragma: no cover - reported, not raised
                with lock:
                    errors.append(type(exc).__name__)
                continue
            with lock:
                durations.append(time.perf_counter() - began)

    parts = [queries[i::threads] for i in range(threads)]
    began = time.perf_counter()
    pool = [threading.Thread(target=worker, args=(part,)) for part in parts]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    wall = time.perf_counter() - began
    return {
        "threads": threads,
        "queries": len(durations),
        "errors": len(errors),
        "throughput_qps": round(len(durations) / wall, 1) if wall else 0.0,
        "latency_p50_ms": round(pct(durations, 50) * 1000, 1),
        "latency_p95_ms": round(pct(durations, 95) * 1000, 1),
    }


def eval_catalog(h: Harness, lib: Library) -> dict:
    counts = h.tools.execute("catalog_counts", {}, admin_scope())
    expected_shelves = {name for book in lib.books.values() if book.page_ids for name in book.shelf_names}
    expected_books = sum(1 for book in lib.books.values() if book.page_ids)
    listed = []
    offset = 0
    calls = 0
    began = time.perf_counter()
    while True:
        result = h.tools.execute("catalog_list_books", {"offset": offset, "limit": 50}, admin_scope())
        calls += 1
        books = result.get("books") or []
        listed.extend(books)
        if len(books) < 50:
            break
        offset += 50
    list_s = time.perf_counter() - began
    sample_allowed = sorted(random.Random(3).sample(sorted(lib.pages), k=min(len(lib.pages), 3000)))
    began = time.perf_counter()
    restricted = h.tools.execute("catalog_counts", {}, user_scope(sample_allowed))
    restricted_ms = (time.perf_counter() - began) * 1000
    exp_restricted_books = len({lib.pages[p].book_id for p in sample_allowed})
    exp_restricted_shelves = len({s for p in sample_allowed for s in lib.pages[p].shelf_names})
    return {
        "pages": {"got": counts.get("pages"), "expected": len(lib.pages)},
        "books": {"got": counts.get("books"), "expected": expected_books},
        "shelves": {"got": counts.get("shelves"), "expected": len(expected_shelves)},
        "list_books_total": {"got": len(listed), "expected": expected_books, "calls_needed": calls, "seconds": round(list_s, 3)},
        "chapters": {"got": counts.get("chapters"), "expected": sum(1 for ch in lib.chapters.values() if ch.page_ids)},
        "list_books_has_shelf_or_chapter_fields": any(("shelf" in k or "chapter" in k) for b in listed[:1] for k in b),
        "restricted_3000": {
            "books": {"got": restricted.get("books"), "expected": exp_restricted_books},
            "shelves": {"got": restricted.get("shelves"), "expected": exp_restricted_shelves},
            "latency_ms": round(restricted_ms, 1),
        },
        "chapter_count_available": "chapters" in counts and counts.get("chapters") == sum(1 for ch in lib.chapters.values() if ch.page_ids),
    }


def eval_hierarchy(h: Harness, lib: Library, samples: int, rng: random.Random) -> dict:
    scope = admin_scope()
    out: Dict[str, dict] = {}

    # 1) Same fact on K pages; the question names the chapter/book/shelf that disambiguates it.
    #    "text": the name is only in the question (tests indexed hierarchy).
    #    "filter": the router also passes the name as a container argument.
    for level, label in (("chapter", "bölümünde"), ("book", "kitabında"), ("shelf", "rafında")):
        for mode in ("text", "filter"):
            correct = top8 = total = 0
            for group in lib.ambiguous:
                for member in group["members"]:
                    name = member[level]
                    query = f"{name} {label} {group['topic']} nedir?"
                    extra = {level: name} if mode == "filter" else {}
                    pages = _page_ids(h.search(query, scope, **extra))
                    total += 1
                    correct += bool(pages[:1] == [member["page_id"]])
                    top8 += member["page_id"] in pages
            k = len(lib.ambiguous[0]["members"]) if lib.ambiguous else 1
            key = f"disambiguate_by_{level}" + ("" if mode == "text" else "_filter")
            out[key] = {
                "queries": total,
                "top1_accuracy": round(correct / max(1, total), 3),
                "in_top8": round(top8 / max(1, total), 3),
                "random_baseline_top1": round(1 / k, 3),
            }

    # 2) Page title questions. Titles never occur in the body.
    ids = rng.sample(sorted(lib.pages), k=min(samples, len(lib.pages)))
    hit = 0
    for page_id in ids:
        pages = _page_ids(h.search(f"{lib.needles[page_id]['title']} sayfasında ne anlatılıyor?", scope))
        hit += page_id in pages
    out["page_title_lookup"] = {"queries": len(ids), "recall_at_8": round(hit / max(1, len(ids)), 3)}

    # 3) Listing questions answered through search: how many passages come from the named container?
    for level in ("shelf", "chapter", "book"):
        precision = []
        coverage = []
        if level == "shelf":
            targets = [s for s in lib.shelves.values() if s.book_ids]
            targets = rng.sample(targets, k=min(samples // 4 or 1, len(targets)))
            for shelf in targets:
                member_pages = {p for b in shelf.book_ids for p in lib.books[b].page_ids}
                passages = h.search(f"{shelf.name} içinde hangi kitaplar ve sayfalar var?", scope)
                pages = _page_ids(passages)
                precision.append(sum(p in member_pages for p in pages) / max(1, len(pages)))
                coverage.append(len({lib.pages[p].book_id for p in pages if p in member_pages}) / max(1, len(shelf.book_ids)))
        elif level == "chapter":
            targets = [c for c in lib.chapters.values() if c.page_ids]
            targets = rng.sample(targets, k=min(samples // 4 or 1, len(targets)))
            for chapter in targets:
                pages = _page_ids(h.search(f"{chapter.name} bölümündeki sayfalar neler?", scope))
                members = set(chapter.page_ids)
                precision.append(sum(p in members for p in pages) / max(1, len(pages)))
                coverage.append(len(set(pages) & members) / max(1, len(members)))
        else:
            targets = [b for b in lib.books.values() if b.page_ids]
            targets = rng.sample(targets, k=min(samples // 4 or 1, len(targets)))
            for book in targets:
                pages = _page_ids(h.search(f"{book.name} kitabında hangi sayfalar var?", scope))
                members = set(book.page_ids)
                precision.append(sum(p in members for p in pages) / max(1, len(pages)))
                coverage.append(len(set(pages) & members) / max(1, len(members)))
        out[f"list_{level}_via_search"] = {
            "queries": len(precision),
            "precision_of_returned_pages": round(statistics.mean(precision), 3) if precision else 0.0,
            "member_coverage": round(statistics.mean(coverage), 3) if coverage else 0.0,
        }

    # 3b) The same listings through catalog_browse: exact, complete answers.
    out["browse"] = eval_browse(h, lib, samples, rng)

    # 4) What the answer model can see about hierarchy.
    passage = (h.search(f"{lib.needles[1]['code']} sisteminin yedekleme sorumlusu kim?", scope) or [{}])[0]
    out["passage_fields"] = sorted(passage.keys())
    out["passage_has_hierarchy"] = any(k in passage for k in ("book_name", "chapter_name", "shelf_names", "shelf_name"))
    chunk = h.store.get_chunks([passage.get("chunk_id", "")]).get(passage.get("chunk_id", ""))
    embed_text = str(chunk["embed_text"]) if chunk else ""
    page = lib.pages[int(passage.get("page_id", 1))]
    fts_row = h.store.conn.execute(
        "SELECT body, title, location FROM chunks_fts WHERE chunk_id = ?", (passage.get("chunk_id", ""),)
    ).fetchone()
    fts_body = "\n".join(str(fts_row[key] or "") for key in ("body", "title", "location")) if fts_row else ""
    out["indexed_text_contains"] = {
        "embedding": {k: v in embed_text for k, v in (("title", page.name), ("book", page.book_name), ("chapter", page.chapter_name), ("shelf", (page.shelf_names or [""])[0]))},
        "fts": {k: v in fts_body for k, v in (("title", page.name), ("book", page.book_name), ("chapter", page.chapter_name), ("shelf", (page.shelf_names or [""])[0]))},
    }
    tool_names = [decl["name"] for decl in FUNCTION_DECLARATIONS]
    out["catalog_tools"] = [n for n in tool_names if n.startswith("catalog")]
    browse = next((decl for decl in FUNCTION_DECLARATIONS if decl["name"] == "catalog_browse"), None)
    levels = set(((browse or {}).get("parameters", {}).get("properties", {}).get("level", {}) or {}).get("enum", []))
    props = set((browse or {}).get("parameters", {}).get("properties", {}))
    out["missing_catalog_capabilities"] = [
        cap for cap, present in (
            ("list_shelves", "shelves" in levels),
            ("list_chapters", "chapters" in levels),
            ("books_in_shelf", "books" in levels and "shelf" in props),
            ("pages_in_book_or_chapter", "pages" in levels and {"book", "chapter"} <= props),
        ) if not present
    ]
    return out


def _browse_all(h: Harness, scope: AuthorizationScope, **args) -> dict:
    items: List[dict] = []
    offset = 0
    calls = 0
    while True:
        result = h.tools.execute("catalog_browse", {**args, "offset": offset, "limit": 50}, scope)
        calls += 1
        if not result.get("ok"):
            return {"items": items, "calls": calls, "error": result.get("error")}
        items.extend(result.get("items") or [])
        if not result.get("has_more"):
            return {"items": items, "calls": calls, "result": result}
        offset += 50


def _fuzzy(name: str) -> str:
    """How a user might type a name: lower case, ASCII only."""
    table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return name.translate(table).lower()


def eval_browse(h: Harness, lib: Library, samples: int, rng: random.Random) -> dict:
    scope = admin_scope()
    out: Dict[str, dict] = {}
    shelves = _browse_all(h, scope, level="shelves")
    expected_shelves = {name for b in lib.books.values() if b.page_ids for name in b.shelf_names}
    out["list_shelves"] = {
        "exact": {item["shelf_name"] for item in shelves["items"]} == expected_shelves,
        "got": len(shelves["items"]), "expected": len(expected_shelves), "calls": shelves["calls"],
    }
    checks = {"books_in_shelf": [], "chapters_in_book": [], "pages_in_chapter": [], "pages_in_book": []}
    for shelf in rng.sample([s for s in lib.shelves.values() if s.book_ids], k=min(10, len(lib.shelves))):
        got = _browse_all(h, scope, level="books", shelf=_fuzzy(shelf.name))
        expected = {b for b in shelf.book_ids if lib.books[b].page_ids}
        checks["books_in_shelf"].append({item["book_id"] for item in got["items"]} == expected)
    books = [b for b in lib.books.values() if b.page_ids]
    for book in rng.sample(books, k=min(20, len(books))):
        got = _browse_all(h, scope, level="chapters", book=_fuzzy(book.name))
        expected = {c for c in book.chapter_ids if lib.chapters[c].page_ids}
        checks["chapters_in_book"].append({item["chapter_id"] for item in got["items"]} == expected)
        got = _browse_all(h, scope, level="pages", book=book.name)
        checks["pages_in_book"].append({item["page_id"] for item in got["items"]} == set(book.page_ids))
    chapters = [c for c in lib.chapters.values() if c.page_ids]
    for chapter in rng.sample(chapters, k=min(20, len(chapters))):
        got = _browse_all(h, scope, level="pages", book=lib.books[chapter.book_id].name, chapter=chapter.name)
        checks["pages_in_chapter"].append({item["page_id"] for item in got["items"]} == set(chapter.page_ids))
    for key, values in checks.items():
        out[key] = {"queries": len(values), "exact": round(sum(values) / max(1, len(values)), 3)}

    # A restricted user must not learn hidden shelf names through listing or name resolution.
    hidden_shelf = next(iter(sorted(expected_shelves)))
    visible_pages = [p for p, page in lib.pages.items() if hidden_shelf not in page.shelf_names]
    user = user_scope(visible_pages)
    listed = {item["shelf_name"] for item in _browse_all(h, user, level="shelves")["items"]}
    resolved = h.tools.execute("catalog_browse", {"level": "books", "shelf": hidden_shelf}, user)
    # `unmatched` echoes the user's own words; only system-provided fields count as a leak.
    exposed = {key: resolved.get(key) for key in ("items", "matched", "suggestions")}
    leaked = hidden_shelf in listed or hidden_shelf in json.dumps(exposed, ensure_ascii=False)
    out["hidden_shelf_leak"] = {"leaked": leaked, "filter_status": resolved.get("filter_status", "")}
    return out


def eval_updates(h: Harness, lib: Library, count: int, rng: random.Random) -> dict:
    """Change content on a full index and check that old facts disappear and new ones appear."""
    scope = admin_scope()
    ids = rng.sample(sorted(lib.pages), k=min(count, len(lib.pages)))
    durations = []
    stale_hits = fresh_hits = 0
    used = {n["code"] for n in lib.needles.values()}
    for page_id in ids:
        old_code = lib.needles[page_id]["code"]
        new_code = _code(rng, used)
        page = lib.pages[page_id]
        updated = page.model_copy(update={"markdown": page.markdown.replace(old_code, new_code), "generation": 2})
        began = time.perf_counter()
        status = h.indexer.upsert(updated, generation=2)
        durations.append(time.perf_counter() - began)
        assert status == "published", status
        lib.pages[page_id] = updated
        lib.needles[page_id]["code"] = new_code
        stale_rows = h.store.lexical_search(fts_match(old_code), 5)
        stale_hits += any(int(row["page_id"]) == page_id for row in stale_rows)
        fresh_hits += page_id in _page_ids(h.search(f"{new_code} sisteminin yedekleme sorumlusu kim?", scope))
    return {
        "updates": len(ids),
        "upsert_p50_ms": round(pct(durations, 50) * 1000, 1),
        "upsert_p95_ms": round(pct(durations, 95) * 1000, 1),
        "old_code_still_lexically_found": stale_hits,
        "new_code_recall_at_8": round(fresh_hits / max(1, len(ids)), 3),
    }


def _dir_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def run(args) -> dict:
    preset = dict(PRESETS[args.preset])
    for key in ("books", "pages", "shelves"):
        if getattr(args, key):
            preset[key] = getattr(args, key)
    checkpoints = sorted({c for c in preset["checkpoints"] if c <= preset["pages"]} | {preset["pages"]})
    root = Path(args.directory or tempfile.mkdtemp(prefix="rag-stress-"))
    rng = random.Random(args.seed)

    began = time.perf_counter()
    lib = build_library(preset["books"], preset["pages"], preset["shelves"], args.sections, args.words, args.seed)
    gen_s = time.perf_counter() - began
    h = Harness(root, args.embedder)

    result: dict = {
        "config": {
            "preset": args.preset, "books": preset["books"], "pages": preset["pages"], "shelves": preset["shelves"],
            "chapters": len(lib.chapters), "sections_per_page": args.sections, "words_per_section": args.words,
            "embedder": args.embedder, "seed": args.seed, "acl_batch": h.settings.acl_batch,
            "generation_seconds": round(gen_s, 2), "directory": str(root),
        },
        "index_windows": [],
        "checkpoints": [],
    }
    window = max(1, preset["pages"] // 10)
    window_start = time.perf_counter()
    index_start = window_start
    for page_id in range(1, preset["pages"] + 1):
        status = h.indexer.upsert(lib.pages[page_id])
        if status != "published":
            raise RuntimeError(f"page {page_id}: {status}")
        if page_id % window == 0:
            elapsed = time.perf_counter() - window_start
            result["index_windows"].append(
                {"pages_indexed": page_id, "window_pages_per_s": round(window / elapsed, 1), "ms_per_page": round(elapsed / window * 1000, 1)}
            )
            print(f"indexed {page_id}/{preset['pages']} ({window / elapsed:.1f} pages/s)", flush=True)
            window_start = time.perf_counter()
        if page_id in checkpoints:
            paused = time.perf_counter()
            cp = {
                "pages": page_id,
                "chunks": h.vectors.collection.count(),
                "needles": eval_needles(h, lib, page_id, args.samples, rng),
            }
            print(f"checkpoint {page_id}: {json.dumps(cp['needles'])}", flush=True)
            result["checkpoints"].append(cp)
            index_start += time.perf_counter() - paused
            window_start += time.perf_counter() - paused
    result["index_total_seconds"] = round(time.perf_counter() - index_start, 1)
    result["storage_mb"] = {"sqlite": _dir_mb(root) - _dir_mb(root / "chroma"), "chroma": _dir_mb(root / "chroma")}

    total = preset["pages"]
    acl_sizes = [s for s in (50, 200, 1000, 3000, 8000) if s < total]
    print("acl...", flush=True)
    result["acl"] = eval_acl(h, lib, total, acl_sizes, max(10, args.samples // 5), rng)
    print("concurrency...", flush=True)
    result["concurrency"] = eval_concurrency(h, lib, total, args.threads, max(5, args.samples // 10), rng)
    print("catalog...", flush=True)
    result["catalog"] = eval_catalog(h, lib)
    print("hierarchy...", flush=True)
    result["hierarchy"] = eval_hierarchy(h, lib, args.samples, rng)
    print("updates...", flush=True)
    result["updates"] = eval_updates(h, lib, args.updates, rng)
    return result


def render_markdown(r: dict) -> str:
    c = r["config"]
    lines = [
        f"# Stres testi sonucu ({c['preset']})",
        "",
        f"Korpus: {c['shelves']} raf, {c['books']} kitap, {c['chapters']} bölüm, {c['pages']} sayfa; "
        f"sayfa başına {c['sections_per_page']} bölüm başlığı × {c['words_per_section']} kelime. Gömme: `{c['embedder']}`.",
        "",
        "## A. Ölçek",
        "",
        f"Toplam indeks süresi: {r['index_total_seconds']} s. Depolama: SQLite {r['storage_mb']['sqlite']} MB, Chroma {r['storage_mb']['chroma']} MB.",
        "",
        "| İndekslenen sayfa | sayfa/s | ms/sayfa |",
        "|---|---|---|",
    ]
    lines += [f"| {w['pages_indexed']} | {w['window_pages_per_s']} | {w['ms_per_page']} |" for w in r["index_windows"]]
    lines += [
        "",
        "| Sayfa | Chunk | Recall@1 | Recall@8 | Yalnız vektör R@8 | Yalnız FTS R@8 | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cp in r["checkpoints"]:
        n = cp["needles"]
        lines.append(
            f"| {cp['pages']} | {cp['chunks']} | {n['recall_at_1']} | {n['recall_at_8']} | {n['vector_only_recall_at_8']} | "
            f"{n['lexical_only_recall_at_8']} | {n['latency_p50_ms']} | {n['latency_p95_ms']} | {n['latency_p99_ms']} |"
        )
    lines += ["", "Kısıtlı kullanıcı (ACL):", "", "| İzinli sayfa | ACL parti | Recall@8 | Sızan sayfa | p50 ms | p95 ms |", "|---|---|---|---|---|---|"]
    lines += [f"| {a['allowed_pages']} | {a['acl_batches']} | {a['recall_at_8']} | {a['leaked_pages']} | {a['latency_p50_ms']} | {a['latency_p95_ms']} |" for a in r["acl"]]
    cc = r["concurrency"]
    lines += ["", f"Eşzamanlılık: {cc['threads']} iş parçacığı, {cc['queries']} sorgu, {cc['errors']} hata, {cc['throughput_qps']} sorgu/s, p50 {cc['latency_p50_ms']} ms, p95 {cc['latency_p95_ms']} ms."]
    u = r["updates"]
    lines += ["", f"Dolu indekste güncelleme: {u['updates']} sayfa, upsert p50 {u['upsert_p50_ms']} ms / p95 {u['upsert_p95_ms']} ms, eski kod bulunan {u['old_code_still_lexically_found']}, yeni kod recall@8 {u['new_code_recall_at_8']}."]
    cat = r["catalog"]
    lines += [
        "",
        "## B. Hiyerarşi",
        "",
        "Katalog aracı (beklenen / dönen):",
        "",
        f"- Sayfa: {cat['pages']['expected']} / {cat['pages']['got']}",
        f"- Kitap: {cat['books']['expected']} / {cat['books']['got']}",
        f"- Raf: {cat['shelves']['expected']} / {cat['shelves']['got']}",
        f"- Kitap listesi: {cat['list_books_total']['expected']} / {cat['list_books_total']['got']} ({cat['list_books_total']['calls_needed']} çağrı)",
        f"- 3000 sayfalık kısıtlı kullanıcı: kitap {cat['restricted_3000']['books']['expected']} / {cat['restricted_3000']['books']['got']}, raf {cat['restricted_3000']['shelves']['expected']} / {cat['restricted_3000']['shelves']['got']}",
        f"- Bölüm: {cat['chapters']['expected']} / {cat['chapters']['got']}",
        f"- Bölüm sayısı doğru dönüyor mu: {cat['chapter_count_available']}; kitap listesinde raf/bölüm alanı: {cat['list_books_has_shelf_or_chapter_fields']}",
        "",
        "| Test | Sorgu | Top-1 | Top-8 | Rastgele top-1 |",
        "|---|---|---|---|---|",
    ]
    hi = r["hierarchy"]
    for level in ("chapter", "book", "shelf"):
        for suffix, label in (("", "ad soruda"), ("_filter", "ad filtre olarak da")):
            d = hi[f"disambiguate_by_{level}{suffix}"]
            lines.append(f"| Aynı bilgi, {level} ile ayırt et ({label}) | {d['queries']} | {d['top1_accuracy']} | {d['in_top8']} | {d['random_baseline_top1']} |")
    lines += ["", f"Sayfa başlığıyla arama recall@8: {hi['page_title_lookup']['recall_at_8']} ({hi['page_title_lookup']['queries']} sorgu).", ""]
    lines += ["| Listeleme sorusu (arama ile) | Sorgu | Dönen sayfaların doğru kapta olma oranı | Kapsama |", "|---|---|---|---|"]
    for level in ("shelf", "book", "chapter"):
        d = hi[f"list_{level}_via_search"]
        lines.append(f"| {level} | {d['queries']} | {d['precision_of_returned_pages']} | {d['member_coverage']} |")
    br = hi["browse"]
    lines += [
        "",
        "catalog_browse ile listeleme (tam doğru oranı):",
        "",
        f"- Raf listesi: {br['list_shelves']['got']} / {br['list_shelves']['expected']}, tam: {br['list_shelves']['exact']}",
    ]
    lines += [f"- {key}: {br[key]['exact']} ({br[key]['queries']} sorgu)" for key in ("books_in_shelf", "chapters_in_book", "pages_in_book", "pages_in_chapter")]
    lines += [f"- Kısıtlı kullanıcıya gizli raf adı sızdı mı: {br['hidden_shelf_leak']['leaked']}"]
    lines += [
        "",
        f"Yanıt modeline giden pasaj alanları: `{', '.join(hi['passage_fields'])}`; hiyerarşi alanı var mı: {hi['passage_has_hierarchy']}.",
        f"İndekslenen metinde bulunanlar: gömme {hi['indexed_text_contains']['embedding']}, FTS {hi['indexed_text_contains']['fts']}.",
        f"Katalog araçları: {hi['catalog_tools']}; eksik yetenekler: {hi['missing_catalog_capabilities']}.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--books", type=int)
    parser.add_argument("--pages", type=int)
    parser.add_argument("--shelves", type=int)
    parser.add_argument("--sections", type=int, default=5)
    parser.add_argument("--words", type=int, default=70)
    parser.add_argument("--samples", type=int, default=200, help="queries per retrieval measurement")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--embedder", choices=("hash-bow", "minilm"), default="hash-bow")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--directory", help="empty directory for the isolated index (default: temp)")
    parser.add_argument("--json", help="write raw results to this path")
    parser.add_argument("--report", help="write a Markdown summary to this path")
    args = parser.parse_args()
    outcome = run(args)
    if args.json:
        Path(args.json).write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(outcome)
    if args.report:
        Path(args.report).write_text(markdown, encoding="utf-8")
    print(markdown)
