"""Build the ChromaDB knowledge base from knowledge_base/sources.yaml.

Each entry in sources.yaml:
  - id: nci_glioma
    title: "..."
    url: "https://..."
    file: sources/nci_glioma.txt      # .txt, .md or .pdf (pdf needs `pip install pypdf`)
    topics: [glioma]                  # optional; used to filter retrieval by tumor type

Usage:  python -m neuroscan.rag.ingest --reset
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from neuroscan.config import load_config, path


def read_text(fp: Path) -> str:
    if fp.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(fp)).pages)
    return fp.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def get_collection(reset: bool = False):
    import chromadb

    cfg = load_config()["rag"]
    client = chromadb.PersistentClient(path=str(path("chroma_dir")))
    if reset:
        try:
            client.delete_collection(cfg["collection"])
        except Exception:  # noqa: BLE001
            pass
    return client.get_or_create_collection(cfg["collection"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    cfg = load_config()["rag"]
    kb = path("knowledge_base")
    sources = yaml.safe_load((kb / "sources.yaml").read_text()) or []
    if not sources:
        raise SystemExit("knowledge_base/sources.yaml lists no sources yet.")

    col = get_collection(reset=args.reset)
    total = 0
    for src in sources:
        fp = kb / src["file"]
        if not fp.exists():
            print(f"missing file for {src['id']}: {fp}")
            continue
        pieces = chunk(read_text(fp), cfg["chunk_chars"], cfg["chunk_overlap"])
        topics = ",".join(src.get("topics", []))
        col.upsert(
            ids=[f"{src['id']}::{i}" for i in range(len(pieces))],
            documents=pieces,
            metadatas=[{"source_id": src["id"], "title": src.get("title", src["id"]),
                        "url": src.get("url", ""), "topics": topics} for _ in pieces],
        )
        total += len(pieces)
        print(f"{src['id']}: {len(pieces)} chunks")
    print(f"Ingested {total} chunks into '{cfg['collection']}'.")


if __name__ == "__main__":
    main()
