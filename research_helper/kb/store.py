"""ChromaDB-backed knowledge base store."""
from __future__ import annotations
import math
from datetime import date
from pathlib import Path

import chromadb
from chromadb.config import Settings

from research_helper.kb import embedder

_KB_DIR = Path("outputs/.kb")
_COLLECTION = "papers"

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _KB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(_KB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


from research_helper.kb._types import KBEntry


def add(
    arxiv_id: str,
    title: str,
    published: str,
    abstract: str,
    report_text: str,
) -> None:
    col = _get_collection()
    docs, ids, metas = [], [], []

    def _stage(suffix: str, text: str, source: str):
        doc_id = f"{arxiv_id}::{suffix}"
        # Skip if already indexed
        existing = col.get(ids=[doc_id])
        if existing["ids"]:
            return
        docs.append(text[:4000])   # cap per chunk to stay within embedding limits
        ids.append(doc_id)
        metas.append({"arxiv_id": arxiv_id, "title": title,
                       "published": published, "source": source})

    _stage("abstract", abstract, "abstract")
    # Index report in 1500-char chunks to improve retrieval granularity
    for i, chunk in enumerate(_chunk(report_text, 1500)):
        _stage(f"report_{i}", chunk, "report")

    if not docs:
        return  # all already indexed

    vectors = embedder.embed(docs)
    col.add(documents=docs, embeddings=vectors, ids=ids, metadatas=metas)


def _recency_score(published: str, arxiv_id: str = "") -> float:
    """Exponential decay from today; half-life ≈ 2 years. Returns (0, 1]."""
    try:
        if published:
            y, m, d = int(published[:4]), int(published[5:7]) or 1, int(published[8:10]) or 1
        else:
            # Infer year/month from arxiv ID format YYMM.xxxxx
            import re
            m_id = re.match(r"(\d{2})(\d{2})\.\d+", arxiv_id or "")
            if not m_id:
                return 0.5
            yy, mm = int(m_id.group(1)), int(m_id.group(2))
            y = 2000 + yy
            m, d = mm, 1
        days_ago = max(0, (date.today() - date(y, m, d)).days)
        return math.exp(-days_ago / 730)
    except Exception:
        return 0.5


def query(text: str, top_k: int = 5, recency_weight: float = 0.3) -> list[KBEntry]:
    col = _get_collection()
    if col.count() == 0:
        return []
    vec = embedder.embed_one(text[:2000])
    # Fetch many more raw chunks than needed so deduplication can yield top_k distinct papers
    fetch = min(top_k * 20, col.count())
    results = col.query(
        query_embeddings=[vec],
        n_results=fetch,
        include=["documents", "metadatas", "distances"],
    )

    # Deduplicate: keep best-similarity chunk per paper
    best: dict[str, tuple[str, dict, float]] = {}
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        aid = meta["arxiv_id"]
        if aid not in best or dist < best[aid][2]:
            best[aid] = (doc, meta, dist)

    # Re-rank by combined score: relevance + recency
    w = max(0.0, min(1.0, recency_weight))
    scored = []
    for aid, (doc, meta, dist) in best.items():
        sim = 1.0 - dist
        recency = _recency_score(meta.get("published", ""), meta.get("arxiv_id", ""))
        combined = sim * (1 - w) + recency * w
        scored.append((combined, doc, meta, dist))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        KBEntry(
            doc_id=f'{meta["arxiv_id"]}::{meta["source"]}',
            text=doc,
            title=meta["title"],
            arxiv_id=meta["arxiv_id"],
            published=meta.get("published", ""),
            source=meta["source"],
            distance=dist,
        )
        for _, doc, meta, dist in scored[:top_k]
    ]


def list_papers() -> list[dict]:
    col = _get_collection()
    if col.count() == 0:
        return []
    results = col.get(where={"source": "abstract"}, include=["metadatas"])
    seen, papers = set(), []
    for meta in results["metadatas"]:
        aid = meta["arxiv_id"]
        if aid not in seen:
            seen.add(aid)
            papers.append({"arxiv_id": aid, "title": meta["title"],
                           "published": meta["published"]})
    papers.sort(key=lambda p: p["published"], reverse=True)
    return papers


def count() -> int:
    return len(list_papers())


def _chunk(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]
