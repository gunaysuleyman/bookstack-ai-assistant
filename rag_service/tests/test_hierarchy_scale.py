"""Hierarchy-aware retrieval, catalog browsing, and scale fixes (offline, hash-bow vectors)."""

import json
import sqlite3
from dataclasses import replace

import pytest

from adaptive.catalog import match_names, normalize
from adaptive.chunking import chunk_document
from adaptive.config import load_settings
from adaptive.contracts import AuthorizationScope, PageDocument
from adaptive.embeddings import HashEmbeddingFunction
from adaptive.hybrid import HybridSearcher
from adaptive.indexer import Indexer
from adaptive.jobs import apply_book_refresh, apply_chapter_refresh, apply_shelf_refresh, container_job_key
from adaptive.store import StateStore
from adaptive.tools import ToolRegistry, resolve_route
from adaptive.vector_index import VectorIndex


def _scope(allowed=None):
    return AuthorizationScope(
        principal="u",
        is_admin=allowed is None,
        can_use_ai=True,
        allowed_page_ids=allowed,
        fingerprint="f",
        acl_version="1",
        issued_at=1,
        expires_at=10**12,
    )


def _page(page_id, name, book_id, book, chapter_id, chapter, shelves, body, generation=1):
    return PageDocument(
        page_id=page_id,
        name=name,
        markdown=body,
        book_id=book_id,
        book_name=book,
        chapter_id=chapter_id,
        chapter_name=chapter,
        shelf_names=shelves,
        url=f"http://wiki/{page_id}",
        generation=generation,
    )


LIBRARY = [
    _page(1, "Seyahat Kuralları", 10, "Finans El Kitabı", 100, "Tedarik Süreçleri", ["Finans Ankara Rafı"],
          "## Parametreler\n\nSeyahat avansı onay limiti 400 TL olarak uygulanır."),
    _page(2, "Avans Notları", 20, "İK El Kitabı", 200, "Bordro İşlemleri", ["İnsan Kaynakları İzmir Rafı"],
          "## Parametreler\n\nSeyahat avansı onay limiti 900 TL olarak uygulanır."),
    _page(3, "Ağ Kurulumu", 30, "BT El Kitabı", 300, "Ağ Altyapısı", ["Bilgi Teknolojileri Ankara Rafı", "Ortak Raf"],
          "## Kurulum\n\nVPN istemcisi portal üzerinden indirilir."),
    _page(4, "Yedekleme", 30, "BT El Kitabı", 0, "General Chapter", ["Bilgi Teknolojileri Ankara Rafı", "Ortak Raf"],
          "## Sorumlu\n\nKX7Q42 sisteminin yedekleme sorumlusu Deniz Aksoy."),
    _page(5, "Gizli Plan", 40, "Yönetim", 400, "Kurul", ["Yönetim Rafı"], "## Plan\n\nMERGER-228 planı kurul rafındadır."),
]


@pytest.fixture()
def rig(tmp_path):
    settings = replace(
        load_settings(), state_dir=str(tmp_path), chroma_dir=str(tmp_path / "chroma"),
        embedding_model_id="hash-bow", answer_mode="extractive",
    )
    store = StateStore(str(tmp_path / "rag_state.sqlite"))
    vectors = VectorIndex(str(tmp_path / "chroma"), "hierarchy_test", HashEmbeddingFunction())
    indexer = Indexer(store, vectors, settings)
    for page in LIBRARY:
        assert indexer.upsert(page) == "published"
    tools = ToolRegistry(store, HybridSearcher(store, vectors, acl_batch=settings.acl_batch), settings)
    yield store, vectors, indexer, tools
    store.close()


def _pages(result):
    return [item["page_id"] for item in result.get("passages") or []]


def test_name_matching_is_case_and_diacritic_insensitive():
    names = ["Finans Ankara Rafı", "Finans İzmir Rafı", "İnsan Kaynakları İzmir Rafı"]
    assert normalize("İnsan KAYNAKLARI") == normalize("insan kaynaklari")
    assert match_names("finans ankara rafi", names) == (["Finans Ankara Rafı"], [])
    assert match_names("Finans Ankara rafındaki", names)[0] == ["Finans Ankara Rafı"]
    # A partial overlap is a suggestion, never a silent substitute.
    matches, suggestions = match_names("Finans Bursa Rafı", names)
    assert matches == [] and "Finans Ankara Rafı" in suggestions


def test_passages_carry_location_and_titles_are_searchable(rig):
    _store, _vectors, _indexer, tools = rig
    result = tools.execute("document_search", {"query": "KX7Q42 yedekleme sorumlusu"}, _scope())
    top = result["passages"][0]
    assert top["page_id"] == 4
    assert top["book_name"] == "BT El Kitabı"
    assert top["chapter_name"] == ""  # placeholder is not shown as a chapter
    assert "Ortak Raf" in top["shelf_names"]
    assert 3 in _pages(tools.execute("document_search", {"query": "Ağ Kurulumu sayfası"}, _scope()))


def test_container_filter_disambiguates_same_fact(rig):
    _store, _vectors, _indexer, tools = rig
    by_chapter = tools.execute("document_search", {"query": "seyahat avansı onay limiti", "chapter": "bordro islemleri"}, _scope())
    assert _pages(by_chapter) == [2]
    assert by_chapter["container"]["status"] == "applied"
    by_shelf = tools.execute("document_search", {"query": "seyahat avansı onay limiti", "shelf": "Finans Ankara Rafı"}, _scope())
    assert _pages(by_shelf) == [1]
    unknown = tools.execute("document_search", {"query": "seyahat avansı onay limiti", "book": "Olmayan Kitap"}, _scope())
    assert unknown["container"]["status"] == "unmatched"
    assert set(_pages(unknown)) >= {1, 2}


def test_catalog_browse_levels_and_counts(rig):
    _store, _vectors, _indexer, tools = rig
    shelves = tools.execute("catalog_browse", {"level": "shelves"}, _scope())
    assert {item["shelf_name"] for item in shelves["items"]} == {
        "Finans Ankara Rafı", "İnsan Kaynakları İzmir Rafı", "Bilgi Teknolojileri Ankara Rafı", "Ortak Raf", "Yönetim Rafı",
    }
    books = tools.execute("catalog_browse", {"level": "books", "shelf": "ortak raf"}, _scope())
    assert [item["book_name"] for item in books["items"]] == ["BT El Kitabı"]
    assert books["items"][0]["chapter_count"] == 1
    chapters = tools.execute("catalog_browse", {"level": "chapters", "book": "bt el kitabi"}, _scope())
    assert [item["chapter_name"] for item in chapters["items"]] == ["Ağ Altyapısı"]
    assert chapters["pages_outside_chapters"] == 1
    pages = tools.execute("catalog_browse", {"level": "pages", "book": "BT El Kitabı"}, _scope())
    assert [item["page_id"] for item in pages["items"]] == [3, 4]
    counts = tools.execute("catalog_counts", {}, _scope())
    assert (counts["pages"], counts["books"], counts["chapters"], counts["shelves"]) == (5, 4, 4, 5)
    in_shelf = tools.execute("catalog_counts", {"shelf": "Bilgi Teknolojileri Ankara Rafı"}, _scope())
    assert (in_shelf["pages"], in_shelf["books"], in_shelf["chapters"]) == (2, 1, 1)


def test_catalog_never_reveals_hidden_containers(rig):
    _store, _vectors, _indexer, tools = rig
    user = _scope([1, 2, 3, 4])
    shelves = tools.execute("catalog_browse", {"level": "shelves"}, user)
    assert "Yönetim Rafı" not in {item["shelf_name"] for item in shelves["items"]}
    hidden = tools.execute("catalog_browse", {"level": "books", "shelf": "Yönetim Rafı"}, user)
    assert hidden["items"] == [] and hidden["filter_status"] == "no_match"
    assert "Yönetim" not in json.dumps({k: hidden.get(k) for k in ("items", "matched", "suggestions")}, ensure_ascii=False)
    assert tools.execute("catalog_counts", {}, user)["books"] == 3
    search = tools.execute("document_search", {"query": "MERGER-228 kurul", "shelf": "Yönetim Rafı"}, user)
    assert 5 not in _pages(search)


def test_route_maps_catalog_list_to_children_of_named_container():
    decision = resolve_route("route", {"intent": "catalog_list", "book": "BT El Kitabı"}, "BT kitabında hangi bölümler var")
    assert decision["tool"] == "catalog_browse"
    assert decision["args"]["level"] == "chapters" and decision["args"]["book"] == "BT El Kitabı"
    decision = resolve_route("route", {"intent": "document", "query": "limit", "chapter": "Bordro"}, "q")
    assert decision["args"]["chapter"] == "Bordro"
    assert resolve_route("catalog_list_books", {}, "q")["validation"] == "accepted"


def test_republish_removes_old_fts_rows_by_rowid(rig):
    store, _vectors, indexer, _tools = rig
    page = LIBRARY[3].model_copy(update={"markdown": "## Sorumlu\n\nZZ9P31 sisteminin yedekleme sorumlusu Ali Kaya.", "generation": 2})
    assert indexer.upsert(page, generation=2) == "published"
    fts_rows = store.conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
    chunk_rows = store.conn.execute("SELECT COUNT(*) AS n FROM chunk_records").fetchone()["n"]
    assert fts_rows == chunk_rows
    assert not store.lexical_search('"KX7Q42"', 5)
    assert store.lexical_search('"ZZ9P31"', 5)
    assert indexer.delete(4, 3) == "tombstone"
    assert not store.lexical_search('"ZZ9P31"', 5)


def test_shelf_change_is_metadata_only_but_rename_reembeds(rig):
    store, _vectors, indexer, tools = rig
    moved = LIBRARY[0].model_copy(update={"shelf_names": ["Ortak Raf"], "generation": 2})
    assert indexer.upsert(moved, generation=2) == "metadata_only"
    books = tools.execute("catalog_browse", {"level": "books", "shelf": "Ortak Raf"}, _scope())
    assert "Finans El Kitabı" in {item["book_name"] for item in books["items"]}
    assert store.lexical_search('"Ortak" AND "avansı"', 5)
    renamed = moved.model_copy(update={"chapter_name": "Masraf Kuralları", "generation": 3})
    assert indexer.upsert(renamed, generation=3) == "published"
    row = store.conn.execute("SELECT embed_text FROM chunk_records WHERE page_id = 1").fetchone()
    assert "Masraf Kuralları" in row["embed_text"]


def test_old_fts_schema_is_migrated(tmp_path):
    path = str(tmp_path / "old.sqlite")
    store = StateStore(path)
    store.close()
    raw = sqlite3.connect(path)
    raw.execute("DROP TABLE chunks_fts")
    raw.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, page_id UNINDEXED, revision_id UNINDEXED, body)")
    raw.execute("INSERT INTO page_state(page_id, status, active_revision, title, book_name, chapter_name, shelf_names) "
                "VALUES (7, 'published', 'r1', 'Eski Sayfa', 'Eski Kitap', 'Eski Bölüm', '[\"Eski Raf\"]')")
    raw.execute("INSERT INTO chunk_records(chunk_id, page_id, revision_id, heading, body) VALUES ('c1', 7, 'r1', 'H', 'gövde metni')")
    raw.commit()
    raw.close()
    store = StateStore(path)
    rows = store.lexical_search('"Raf"', 5)
    assert [row["chunk_id"] for row in rows] == ["c1"]
    assert store.conn.execute("SELECT fts_rowid FROM chunk_records WHERE chunk_id = 'c1'").fetchone()["fts_rowid"] is not None
    store.close()


def test_large_acl_uses_single_queries_and_never_leaks(rig):
    store, _vectors, _indexer, tools = rig
    allowed = [1, 2, 3, 4] + list(range(1000, 1500))  # many unknown ids push the list past one ACL batch
    result = tools.execute("document_search", {"query": "MERGER-228 kurul plan"}, _scope(allowed))
    assert 5 not in _pages(result)
    assert all(row["page_id"] != "5" for row in store.lexical_search('"MERGER"', 5, page_ids=allowed))


def test_sentences_are_not_cut_between_chunks():
    filler = " ".join(["uzunkelimelerdenoluşanbirdolgu"] * 40) + "."
    page = PageDocument(page_id=1, name="P", markdown=f"{filler} KX7Q42 sisteminin yedekleme sorumlusu Deniz.")
    _parents, children = chunk_document(page, child_tokens=140, embed_max_tokens=180)
    assert any("KX7Q42 sisteminin yedekleme sorumlusu Deniz." in child.text for child in children)


def test_container_jobs_do_not_collide_with_page_jobs(tmp_path):
    store = StateStore(str(tmp_path / "jobs.sqlite"))
    page_job = store.enqueue(5, "page_upsert", {"page_id": 5})
    store.enqueue(container_job_key("book", 5), "book_refresh", {"book_id": 5})
    store.enqueue(container_job_key("chapter", 5), "chapter_refresh", {"chapter_id": 5})
    status = store.conn.execute("SELECT status FROM sync_jobs WHERE id = ?", (page_job,)).fetchone()["status"]
    assert status == "queued"
    assert len({container_job_key(kind, 5) for kind in ("book", "chapter", "shelf")} | {5}) == 4
    store.close()


def test_container_refresh_jobs(rig):
    store, _vectors, _indexer, _tools = rig
    queued = []
    shelf_only = apply_book_refresh(store, 30, lambda _id: {"book_name": "BT El Kitabı", "shelf_names": ["Yeni Raf"]}, queued.append)
    assert shelf_only["action"] == "relabelled" and queued == []
    assert store.catalog_book_ids_for_shelf("Yeni Raf") == [30]
    renamed = apply_book_refresh(store, 30, lambda _id: {"book_name": "BT Kılavuzu", "shelf_names": ["Yeni Raf"]}, queued.append)
    assert renamed["reason"] == "book_renamed" and sorted(queued) == [3, 4]
    queued.clear()
    apply_book_refresh(store, 40, lambda _id: None, queued.append)
    assert queued == [5]
    queued.clear()
    apply_chapter_refresh(store, 100, lambda _id: {"page_ids": [1, 9]}, queued.append)
    assert sorted(queued) == [1, 9]
    # Book 30 moved to "Yeni Raf" above; the shelf job refreshes books listed
    # by the API and books still indexed under the shelf's name.
    books = []
    apply_shelf_refresh(store, 1, "Yeni Raf", lambda _id: {"name": "Yeni Raf", "book_ids": [10]}, books.append)
    assert sorted(books) == [10, 30]


def _usage():
    from adaptive.provider import LLMUsage

    return LLMUsage(
        provider="fake", model="fake", purpose="test", prompt_tokens_est=1, completion_tokens_est=1,
        prompt_tokens_actual=None, completion_tokens_actual=None, latency_ms=1.0, attempts=1,
    )


def test_engine_routes_catalog_and_keeps_container_on_follow_up(rig, tmp_path):
    from adaptive.engine import AdaptiveEngine
    from adaptive.provider import FunctionCall, LLMResult

    store, vectors, _indexer, _tools = rig
    settings = replace(
        load_settings(), state_dir=str(tmp_path), chroma_dir=str(tmp_path / "chroma"),
        embedding_model_id="hash-bow", answer_mode="llm", tools_enabled=True,
    )
    responses = []

    def fake_llm(route_args, judge):
        def llm(system, prompt, purpose, **kwargs):
            if purpose == "tool_select":
                return LLMResult("", _usage(), [FunctionCall(name="route", args=route_args, call_id="c1")])
            if kwargs.get("contents") and kwargs.get("tool_config"):
                responses.append(kwargs["contents"][-1]["parts"][0]["functionResponse"]["response"])
                return LLMResult("cevap", _usage(), [])
            return LLMResult(json.dumps(judge(prompt)), _usage(), [])
        return llm

    engine = AdaptiveEngine(store, vectors, settings, llm=fake_llm({"intent": "catalog_list", "shelf": "ortak raf"}, lambda _p: {}))
    engine.answer("Ortak rafta hangi kitaplar var?", _scope())
    assert responses[-1]["level"] == "books"
    assert [item["book_name"] for item in responses[-1]["items"]] == ["BT El Kitabı"]

    executed = []
    original = engine.tools.execute

    def spy(name, args, scope, current_page=None):
        executed.append((name, dict(args)))
        return original(name, args, scope, current_page=current_page)

    engine.tools.execute = spy

    def judge(prompt):
        ids = [json.loads(line)["chunk_id"] for line in prompt.splitlines() if line.startswith("{")]
        assert all("Bordro İşlemleri" in json.loads(line)["location"] for line in prompt.splitlines() if line.startswith("{"))
        return {"separate_topics": False, "needs": ["limit"], "unmet_needs": ["istisna"], "use_chunk_ids": ids[:1],
                "coverage": "partial", "ambiguity": "", "followup_query": "avans istisnaları"}

    engine.llm = fake_llm({"intent": "document", "query": "seyahat avansı onay limiti", "chapter": "Bordro İşlemleri"}, judge)
    engine.answer("Bordro İşlemleri bölümünde seyahat avansı onay limiti nedir?", _scope())
    searches = [args for name, args in executed if name == "document_search"]
    assert len(searches) == 2 and all(args.get("chapter") == "Bordro İşlemleri" for args in searches)
