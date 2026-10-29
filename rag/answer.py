"""Question answering over the handbook: retrieve -> answer with citations -> validate.

Guarantees the caller can rely on (checked in code, not just asked of the model):
- every non-refused answer cites at least one passage that was actually retrieved;
- weak retrieval (best cosine below MIN_COSINE) is refused before any model call;
- capability numbers (Cp/Cpk) come from rag.tools, never from the model's arithmetic.

Usage: python -m rag.answer "What is the difference between Cp and Cpk?"
"""
import json
import re
import sys

from . import llm
from .retrieve import search
from .tools import CAPABILITY_SCHEMA, TOOLS

MIN_COSINE = 0.58   # set below the lowest answerable score in eval (0.625); borderline cases go to the model
TOP_K = 5
REFUSAL = "The handbook passages I can search do not cover this, so I won't guess."

SYSTEM = (
    "You answer questions about statistical process control and design of experiments using ONLY the "
    "numbered handbook passages provided. Rules: (1) Every claim must come from the passages; cite the "
    "passage ids you used. (2) If the passages do not contain the answer, set refused=true and leave "
    "citations empty. (3) You have one tool, process_capability, which you call by filling the tool_call "
    "field of your JSON reply (it is always available this way). If the user gives measurements and spec "
    "limits and asks for Cp/Cpk/capability, do not calculate yourself: fill tool_call; you will then receive "
    "the tool result and write the final answer. (4) Answer in the user's language, "
    "concisely (at most 120 words). Reply with JSON only."
)

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "citations", "refused", "tool_call"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "refused": {"type": "boolean"},
        "tool_call": {
            "anyOf": [
                {"type": "null"},
                {"type": "object", "additionalProperties": False, "required": ["name", "args"],
                 "properties": {"name": {"type": "string", "enum": [CAPABILITY_SCHEMA["name"]]},
                                "args": {"type": "object", "additionalProperties": False,
                                         "required": ["values", "lsl", "usl"],
                                         "properties": {"values": {"type": "array", "items": {"type": "number"}},
                                                        "lsl": {"type": "number"},
                                                        "usl": {"type": "number"}}}}},
            ]
        },
    },
}


def build_prompt(question, hits, tool_result=None):
    parts = [f"[{h['id']}] {h['title']}\n{h['text']}" for h in hits]
    prompt = "Passages:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}\n"
    prompt += f"\nAvailable tool: {json.dumps(CAPABILITY_SCHEMA)}\n"
    if tool_result is not None:
        prompt += (f"\nTool result (authoritative, use these numbers exactly): {json.dumps(tool_result)}\n"
                   "Now write the final answer; tool_call must be null.\n")
    return prompt


def parse(raw):
    """Accept either a dict (structured output) or text that contains one JSON object."""
    if isinstance(raw, dict):
        return raw
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("model reply has no JSON object")
    return json.loads(m.group(0))


def validate(reply, hit_ids):
    """Return a list of problems; empty means the reply can be shown."""
    problems = []
    for key in ("answer", "citations", "refused"):
        if key not in reply:
            problems.append(f"missing {key}")
    if problems:
        return problems
    bad = [c for c in reply["citations"] if c not in hit_ids]
    if bad:
        problems.append(f"cites passages that were not retrieved: {bad}")
    if not reply["refused"] and not reply["citations"] and not reply.get("tool_call"):
        problems.append("answer has no citation")
    return problems


def ask(question, backend=None, complete=None):
    """Answer one question. `complete` can be injected for tests (same signature as llm.complete)."""
    complete = complete or llm.complete
    hits = search(question, k=TOP_K)
    trace = {"hits": [{k: h[k] for k in ("id", "title", "url", "cosine")} for h in hits], "llm_calls": 0}
    if not hits or max(h["cosine"] for h in hits) < MIN_COSINE:
        return {"answer": REFUSAL, "citations": [], "refused": True, "tool_result": None,
                "gate": "low_retrieval", **trace}
    ids = {h["id"] for h in hits}

    def call(prompt):
        trace["llm_calls"] += 1
        return parse(complete(prompt, SYSTEM, backend=backend, schema=ANSWER_SCHEMA))

    prompt = build_prompt(question, hits)
    reply = call(prompt)
    problems = validate(reply, ids)
    if problems:  # one repair attempt with the concrete problem stated
        reply = call(prompt + f"\nYour previous reply was rejected: {problems}. Fix it.\n")
        problems = validate(reply, ids)
    tool_result = None
    if not problems and reply.get("tool_call"):
        tc = reply["tool_call"]
        try:
            tool_result = TOOLS[tc["name"]](**tc["args"])
        except (KeyError, TypeError, ValueError) as e:
            tool_result = {"error": str(e)}
        reply = call(build_prompt(question, hits, tool_result))
        problems = validate(reply, ids)
    if problems:
        return {"answer": REFUSAL, "citations": [], "refused": True, "tool_result": tool_result,
                "gate": f"invalid_reply: {problems}", **trace}
    return {"answer": reply["answer"], "citations": reply["citations"], "refused": reply["refused"],
            "tool_result": tool_result, "gate": None, **trace}


if __name__ == "__main__":
    out = ask(" ".join(sys.argv[1:]))
    print(out["answer"])
    for cid in out["citations"]:
        hit = next(h for h in out["hits"] if h["id"] == cid)
        print(f"  [{cid}] {hit['title']}  {hit['url']}")
    if out["tool_result"]:
        print("  tool:", out["tool_result"])
