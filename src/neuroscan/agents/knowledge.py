"""Knowledge agent: RAG over the curated knowledge base -> short cited background note.

With an LLM: the model writes 2-4 sentences, each ending with a citation like [1].
Without an LLM: falls back to quoting sentences verbatim from the top retrieved chunks
(extractive mode), which is grounded by construction.
Skipped when the classifier predicts no tumor.
"""
from __future__ import annotations

import re

from neuroscan import llm
from neuroscan.agents.state import NeuroScanState
from neuroscan.config import load_config
from neuroscan.rag.retriever import retrieve

READABLE = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary tumor"}

SYSTEM = (
    "You write short, neutral background notes for radiology reviewers. Use ONLY the numbered "
    "excerpts provided. Do not give a diagnosis, prognosis or treatment advice for this patient."
)


def _prompt(tumor: str, sources: list[dict]) -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {s['title']}\n{s['text']}" for i, s in enumerate(sources))
    return (
        f"Excerpts:\n\n{excerpts}\n\n"
        f"Write 2 to 4 sentences of general background about {tumor}s (what they are, where they "
        f"arise, typical imaging appearance if stated). End EVERY sentence with exactly one citation "
        f"like [1] pointing to the excerpt that supports it. Output only the sentences."
    )


def split_cited(text: str) -> list[dict]:
    out = []
    for sent in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", text.strip()):
        m = re.search(r"\[(\d+)\]\s*\.?\s*$", sent.strip())
        clean = re.sub(r"\s*\[\d+\]", "", sent).strip()
        if clean:
            out.append({"text": clean, "citation": int(m.group(1)) if m else None})
    return out


def _extractive(sources: list[dict], n: int = 3) -> list[dict]:
    sents = []
    for i, s in enumerate(sources):
        for sent in re.split(r"(?<=[.!?])\s+", s["text"]):
            if 40 <= len(sent) <= 300:
                sents.append({"text": sent.strip(), "citation": i + 1})
                break
        if len(sents) == n:
            break
    return sents


def knowledge_agent(state: NeuroScanState) -> dict:
    cls = state.get("classification") or {}
    label = cls.get("label")
    no_tumor = load_config()["no_tumor_class"]
    if not label or label == no_tumor:
        return {"knowledge": {"sentences": [], "sources": [], "mode": "skipped", "removed": []}}

    tumor = READABLE.get(label, label)
    sources = retrieve(f"{tumor} brain tumor overview MRI appearance", topic=label)
    if not sources:
        return {"knowledge": {"sentences": [], "sources": [], "mode": "skipped", "removed": []}}

    errors, mode = [], "extractive"
    sentences = []
    if llm.available():
        try:
            sentences = split_cited(llm.complete(_prompt(tumor, sources), system=SYSTEM))
            mode = "llm"
        except Exception as e:  # noqa: BLE001
            errors.append(f"knowledge: LLM call failed ({e}); using extractive fallback")
    if not sentences:
        sentences = _extractive(sources)
        mode = "extractive"

    for s in sentences:
        c = s.get("citation")
        s["chunk_id"] = sources[c - 1]["chunk_id"] if c and 1 <= c <= len(sources) else None
        s["grounded"] = None
    out = {"knowledge": {"sentences": sentences, "sources": sources, "mode": mode, "removed": []}}
    if errors:
        out["errors"] = errors
    return out
