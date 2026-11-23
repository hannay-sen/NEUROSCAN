"""Hallucination check for the knowledge agent.

Step 1 (generate): write each explanation sentence, its cited source and the judge's
verdict to results/grounding_labels.csv, with an empty `human_supported` column.
Step 2: open the CSV and fill human_supported with 1 (supported) or 0 (not supported).
Step 3 (summarize): hallucination rate before/after the judge, and judge vs. human agreement.

  python -m neuroscan.eval.evaluate_grounding generate --repeats 5
  python -m neuroscan.eval.evaluate_grounding summarize
"""
from __future__ import annotations

import argparse

import pandas as pd

from neuroscan.agents.knowledge import knowledge_agent
from neuroscan.agents.safety import grounding_review
from neuroscan.config import load_config, path
from neuroscan.eval.common import save_json


def generate(repeats: int) -> None:
    cfg = load_config()
    rows = []
    for label in [c for c in cfg["classes"] if c != cfg["no_tumor_class"]]:
        for r in range(repeats):
            know = knowledge_agent({"classification": {"label": label}})["knowledge"]
            if not know["sentences"]:
                print(f"{label}: no sentences (empty knowledge base or no LLM?)")
                break
            kept, removed, _ = grounding_review(know)
            src = {s["chunk_id"]: s for s in know["sources"]}
            for s in kept + removed:
                rows.append({"tumor": label, "run": r, "mode": know["mode"], "sentence": s["text"],
                             "source_title": src.get(s.get("chunk_id"), {}).get("title", ""),
                             "source_text": src.get(s.get("chunk_id"), {}).get("text", ""),
                             "judge_supported": int(bool(s["grounded"])), "human_supported": ""})
    out = path("results") / "grounding_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} sentences -> {out}. Fill `human_supported` with 1/0, then run summarize.")


def summarize() -> None:
    df = pd.read_csv(path("results") / "grounding_labels.csv")
    df = df[df.human_supported.astype(str).isin(["0", "1", "0.0", "1.0"])].copy()
    if df.empty:
        raise SystemExit("No human labels yet.")
    df["human_supported"] = df.human_supported.astype(float).astype(int)
    kept = df[df.judge_supported == 1]
    res = {
        "n_labeled": len(df),
        "hallucination_rate_before_judge": float(1 - df.human_supported.mean()),
        "hallucination_rate_after_judge": float(1 - kept.human_supported.mean()) if len(kept) else None,
        "judge_human_agreement": float((df.judge_supported == df.human_supported).mean()),
        "unsupported_caught_by_judge": float(
            ((df.judge_supported == 0) & (df.human_supported == 0)).sum() / max((df.human_supported == 0).sum(), 1)),
        "supported_wrongly_removed": float(
            ((df.judge_supported == 0) & (df.human_supported == 1)).sum() / max((df.human_supported == 1).sum(), 1)),
    }
    for k, v in res.items():
        print(f"{k:34s} {v}")
    save_json("grounding_summary", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generate", "summarize"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    generate(args.repeats) if args.step == "generate" else summarize()


if __name__ == "__main__":
    main()
