"""Query the knowledge base. Returns [] if ChromaDB or the collection is unavailable."""
from __future__ import annotations

from neuroscan.config import load_config, path


def retrieve(query: str, topic: str | None = None, k: int | None = None) -> list[dict]:
    cfg = load_config()["rag"]
    k = k or cfg["top_k"]
    if not path("chroma_dir").exists():
        return []
    try:
        from neuroscan.rag.ingest import get_collection

        col = get_collection()
        n = col.count()
        if n == 0:
            return []
        res = col.query(query_texts=[query], n_results=min(k * 3, n))
    except Exception:  # noqa: BLE001
        return []

    hits = []
    for doc_id, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                       res["metadatas"][0], res["distances"][0]):
        topics = [t for t in (meta.get("topics") or "").split(",") if t]
        if topic and topics and topic not in topics:
            continue
        hits.append({"chunk_id": doc_id, "text": doc, "title": meta.get("title", ""),
                     "url": meta.get("url", ""), "distance": float(dist)})
        if len(hits) == k:
            break
    return hits
