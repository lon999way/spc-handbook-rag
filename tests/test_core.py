"""Offline tests: no model calls. The model is replaced by a scripted fake `complete`."""
import pytest

from rag import answer
from rag.tools import process_capability


def test_capability_matches_hand_calculation():
    # mean 50.2, s = 0.4 is awkward to hit exactly, so check against the textbook formulas directly
    vals = [49.8, 50.2, 50.6, 50.0, 50.4]
    r = process_capability(vals, 48, 52)
    mean = sum(vals) / 5
    s = (sum((v - mean) ** 2 for v in vals) / 4) ** 0.5
    assert r["cp"] == round(4 / (6 * s), 3)
    assert r["cpk"] == round(min(52 - mean, mean - 48) / (3 * s), 3)


@pytest.mark.parametrize("vals,lsl,usl", [([1.0], 0, 2), ([1, 2], 3, 1), ([5, 5, 5], 0, 10)])
def test_capability_rejects_bad_input(vals, lsl, usl):
    with pytest.raises(ValueError):
        process_capability(vals, lsl, usl)


def test_validate_flags_uncited_and_foreign_citations():
    ids = {"c0001", "c0002"}
    assert answer.validate({"answer": "x", "citations": ["c0001"], "refused": False}, ids) == []
    assert answer.validate({"answer": "x", "citations": [], "refused": False}, ids)
    assert answer.validate({"answer": "x", "citations": ["c9999"], "refused": False}, ids)
    assert answer.validate({"answer": "", "citations": [], "refused": True}, ids) == []


def test_parse_accepts_fenced_json():
    assert answer.parse('```json\n{"a": 1}\n```') == {"a": 1}


def fake_hits(cos=0.8):
    return [{"id": "c0001", "title": "6.1.6. What is Process Capability?", "url": "u", "text": "t",
             "cosine": cos, "score": 0.1}]


def test_low_retrieval_refuses_without_model_call(monkeypatch):
    monkeypatch.setattr(answer, "search", lambda q, k: fake_hits(cos=0.3))
    calls = []
    out = answer.ask("What was Micron's revenue?", complete=lambda *a, **k: calls.append(1))
    assert out["refused"] and out["gate"] == "low_retrieval" and calls == []


def test_tool_numbers_come_from_tool_not_model(monkeypatch):
    monkeypatch.setattr(answer, "search", lambda q, k: fake_hits())
    replies = iter([
        {"answer": "", "citations": [], "refused": False,
         "tool_call": {"name": "process_capability", "args": {"values": [9.8, 10.2, 10.0, 10.4], "lsl": 9, "usl": 11}}},
        {"answer": "Cpk is 0.88.", "citations": ["c0001"], "refused": False, "tool_call": None},
    ])
    out = answer.ask("Cpk for 9.8, 10.2, 10.0, 10.4 with spec 9-11?", complete=lambda *a, **k: next(replies))
    assert out["tool_result"] == process_capability([9.8, 10.2, 10.0, 10.4], 9, 11)
    assert out["llm_calls"] == 2 and out["citations"] == ["c0001"]


def test_invalid_reply_twice_becomes_refusal(monkeypatch):
    monkeypatch.setattr(answer, "search", lambda q, k: fake_hits())
    bad = {"answer": "made up", "citations": ["c9999"], "refused": False, "tool_call": None}
    out = answer.ask("q", complete=lambda *a, **k: dict(bad))
    assert out["refused"] and out["gate"].startswith("invalid_reply") and out["llm_calls"] == 2
