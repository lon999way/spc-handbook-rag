"""Evaluate the assistant on eval/questions.jsonl.

  python -m eval.run_eval retrieval     # no model calls: hit@5 and refusal-gate accuracy
  python -m eval.run_eval answers       # RAG vs. no-retrieval baseline (needs a model backend)

Scoring is deliberately simple and transparent:
- answer correct  = every expected keyword appears in the answer (string match; a human should still
                    read a sample, string matching can miss paraphrases or pass a wrong sentence);
- citation correct = at least one cited passage comes from an expected handbook page;
- refusal correct = unanswerable questions are refused, answerable ones are not;
- tool exact      = the Cpk in the answer equals the value computed by rag.tools.
Results are written to eval/results_<mode>.json.
"""
import json
import re
import sys
from pathlib import Path

from rag import answer, llm
from rag.retrieve import search
from rag.tools import process_capability

HERE = Path(__file__).resolve().parent
QUESTIONS = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]

BASELINE_SYSTEM = (
    "Answer questions about statistical process control and design of experiments from your own knowledge. "
    "If you do not know or the question needs information you do not have, say you cannot answer. "
    "Be concise (at most 120 words)."
)


def retrieval():
    rows = []
    for q in QUESTIONS:
        hits = search(q["question"], k=answer.TOP_K)
        best = max(h["cosine"] for h in hits)
        rows.append({"id": q["id"], "type": q["type"], "best_cosine": best,
                     "hit": any(h["url"] in q.get("expect_urls", []) for h in hits),
                     "top_urls": [h["url"] for h in hits]})
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    summary = {
        "n": len(rows),
        "hit_at_5": f"{sum(r['hit'] for r in ans)}/{len(ans)}",
        "gate_threshold": answer.MIN_COSINE,
        "gate_refuses_unanswerable": f"{sum(r['best_cosine'] < answer.MIN_COSINE for r in una)}/{len(una)}",
        "gate_false_refusals": f"{sum(r['best_cosine'] < answer.MIN_COSINE for r in ans)}/{len(ans)}",
        "cosine_range_answerable": [min(r["best_cosine"] for r in ans), max(r["best_cosine"] for r in ans)],
        "cosine_range_unanswerable": [min(r["best_cosine"] for r in una), max(r["best_cosine"] for r in una)],
    }
    return summary, rows


def keywords_ok(text, q):
    low = text.lower()
    return all(k.lower() in low for k in q.get("expect_keywords", []))


def cpk_in(text, value):
    # tolerance 0.0051: an answer rounded to 2 decimals (e.g. 2.38 for 2.378) counts as correct
    return any(abs(float(x) - value) < 0.0051 for x in re.findall(r"-?\d+\.\d+", text))


def answers(only=None):
    rows = []
    for q in QUESTIONS:
        if only and q["type"] != only:
            continue
        rag = answer.ask(q["question"])
        urls = {h["id"]: h["url"] for h in rag["hits"]}
        base = llm.complete(q["question"], BASELINE_SYSTEM)
        row = {"id": q["id"], "type": q["type"], "rag_answer": rag["answer"], "rag_refused": rag["refused"],
               "rag_citations": [urls[c] for c in rag["citations"]], "baseline_answer": base}
        if q["type"] == "unanswerable":
            row["rag_ok"] = rag["refused"]
            row["baseline_ok"] = bool(re.search(r"cannot|can't|don't have|do not have|not able|unable|no access",
                                                base, re.I))
        elif q["type"] == "tool":
            want = process_capability(**q["tool_args"])["cpk"]
            row["rag_ok"] = not rag["refused"] and rag["tool_result"] is not None and cpk_in(rag["answer"], want)
            row["baseline_ok"] = cpk_in(base, want)
            row["expected_cpk"] = want
        else:
            row["rag_ok"] = not rag["refused"] and keywords_ok(rag["answer"], q)
            row["baseline_ok"] = keywords_ok(base, q)
            row["rag_citation_ok"] = any(u in q["expect_urls"] for u in row["rag_citations"])
        rows.append(row)
        print(f"{q['id']} rag={row['rag_ok']} base={row['baseline_ok']}", flush=True)

    def rate(key, kind):
        sel = [r for r in rows if r["type"] == kind]
        return f"{sum(bool(r.get(key)) for r in sel)}/{len(sel)}"

    kinds = [k for k in ("answerable", "unanswerable", "tool") if any(r["type"] == k for r in rows)]
    summary = {k: {"rag": rate("rag_ok", k), "baseline": rate("baseline_ok", k)} for k in kinds}
    if "answerable" in summary:
        summary["answerable"]["rag_citation_correct"] = rate("rag_citation_ok", "answerable")
        summary["answerable"]["baseline_citation_correct"] = "0/{} (baseline has no sources)".format(
            sum(r["type"] == "answerable" for r in rows))
    return summary, rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    only = sys.argv[2] if len(sys.argv) > 2 else None  # e.g. "tool" to rerun one question type
    summary, rows = retrieval() if mode == "retrieval" else answers(only)
    suffix = f"_{only}" if only else ""
    (HERE / f"results_{mode}{suffix}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                                          ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
