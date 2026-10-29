# SPC & DOE Handbook Assistant

A small retrieval-augmented (RAG) assistant that answers process-control and design-of-experiments
questions **only** from the NIST/SEMATECH *e-Handbook of Statistical Methods* (chapter 5, Process
Improvement, and chapter 6, Process or Product Monitoring and Control), and cites the handbook pages it used.

It is built the way I would want an engineering assistant to behave in a fab:

- **Cites or refuses.** Every answer must cite retrieved passages; citations are checked in code. If
  retrieval is weak, the assistant refuses before calling the model instead of guessing.
- **Code does the arithmetic.** For Cp / Cpk questions the model requests a `process_capability` tool;
  the numbers shown come from Python, not from the language model.
- **Measured, not assumed.** An evaluation set compares the assistant with a no-retrieval baseline
  (see [Results](#results)).

## How it works

```
question ─► hybrid retrieval (BM25 + bge-small embeddings, reciprocal rank fusion)
         ─► gate: best cosine < threshold → refuse (no model call)
         ─► model reply as JSON {answer, citations, refused, tool_call}  (schema-enforced)
         ─► validate: citations ⊆ retrieved ids, non-refused answers must cite  (1 repair retry)
         ─► tool_call? run rag.tools.process_capability → model writes final answer with exact numbers
```

| File | Role |
|---|---|
| `rag/crawl.py` | polite crawler for handbook chapters 5–6 (1 request/second) |
| `rag/chunk.py` | HTML → ~180-word chunks with page title and URL |
| `rag/retrieve.py` | BM25 + dense embeddings (fastembed, runs on CPU), rank fusion |
| `rag/answer.py` | prompt, schema, validation, refusal gate, tool loop |
| `rag/tools.py` | `process_capability` (Cp, Cpk, mean, sample s) |
| `rag/llm.py` | model backends: local `claude -p` (default) or Anthropic API |
| `eval/` | 30-question evaluation set and scorer |
| `tests/` | offline tests with a scripted fake model |
| `app.py` | Streamlit UI |

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m rag.crawl && .venv/bin/python -m rag.chunk
.venv/bin/python -m rag.retrieve "what is a CUSUM chart"     # builds the embedding cache on first run
.venv/bin/python -m pytest -q tests
.venv/bin/streamlit run app.py
```

Model backend: by default the local Claude Code CLI (`claude -p`, logged in). To use the API instead,
set `RAG_BACKEND=anthropic` and `ANTHROPIC_API_KEY` in your environment.

## Results

Evaluation set: 30 questions (`eval/questions.jsonl`) — 20 answerable from the handbook, 5 capability
calculations, 5 out of scope (company-specific or unrelated). Every keyword used for scoring was first
checked to appear on the expected handbook page.

**Retrieval (no model calls)** — `python -m eval.run_eval retrieval`

| Metric | Result |
|---|---|
| Expected handbook page in top 5 (answerable + tool questions) | 24/25 |
| Out-of-scope questions refused by the retrieval gate | 3/5 (the other 2 fall back to the model's refusal rule) |
| Answerable questions wrongly refused by the gate | 0/25 |
| Best cosine, answerable vs. out of scope | 0.625–0.884 vs. 0.493–0.611 |

The one miss (a17, "why add center points") ranks the central-composite-design pages above the
center-points page. The gate threshold (0.58) sits below the lowest answerable score instead of
exactly between the two groups, so borderline questions reach the model rather than being refused.

Two fixes came out of this evaluation: handbook formulas are MathJax LaTeX (`\hat{C}_{pk}`), so a plain
"Cpk" query initially missed the capability page and was refused; normalizing LaTeX for both keyword
and embedding search raised that question from 0.538 to above the gate, and table-of-contents pages
were removed from the index because they matched every query.

**Answers, RAG vs. no-retrieval baseline** — `python -m eval.run_eval answers`

Run on 2026-10-29 with Claude Haiku 4.5 through the local Claude Code CLI; baseline = same model, same
question, no retrieval.

| Question type | Metric | RAG | Baseline |
|---|---|---|---|
| Answerable (20) | keyword score | 18/20 | 15/20 |
| Answerable (20) | cites the expected handbook page | 19/20 | — (no sources) |
| Out of scope (5) | refused | 5/5 | 5/5 |
| Capability (5) | Cpk correct (±0.005) | 4/5 → **5/5** after fix | 5/5 |

What I learned from reading every answer, not just the scores:

- **On textbook questions the plain model was already accurate.** I read the opening of all 40 answers and
  the full text of every disagreement and found no clear factual errors on either side; the keyword gap
  (18 vs 15) comes from wording ("p-chart" vs "p chart", "axial" vs "star points"), not from wrong answers.
  The baseline does add advice the handbook does not give (e.g. "stop production immediately" for an
  out-of-control process, where the handbook points to the out-of-control action plan). One question (a06, "count of defects per unit") was
  ambiguous in my own wording: RAG answered u chart, the baseline c chart, and both are defensible.
- **The difference is traceability.** RAG answers point to the exact handbook page in 19/20 cases, so an
  engineer can check them; the baseline gives no way to verify where an answer came from.
- **Tool failure found and fixed.** In the first run the model once replied that the capability tool was
  "not available" instead of requesting it (t02). The system prompt now states how the tool is called;
  a rerun of the 5 capability questions gave 5/5 with every number coming from `rag.tools`. (This rerun
  is on the same questions used to find the bug, so it confirms the fix rather than measuring it fresh.)
- **The plain model also computed Cpk correctly** on these small samples (within rounding). The tool's
  value here is exactness and reproducibility, not a large accuracy gain.

Raw outputs: `eval/results_answers_v1.json` (first full run), `eval/results_answers_tool.json` (capability
rerun after the fix), `eval/results_retrieval.json`.

## Limitations

- Many handbook formulas are images, so they are not in the indexed text; answers about a formula may
  describe it in words only.
- Keyword-based answer scoring can miss correct paraphrases or pass a wrong sentence that happens to
  contain the keywords; I read the scored answers by hand as well.
- 30 questions is a small evaluation set, and the refusal threshold was calibrated on the same set.

## Source and license

Content: *NIST/SEMATECH e-Handbook of Statistical Methods*, https://doi.org/10.18434/M32189 — public
information published by NIST (see https://www.nist.gov/copyrights-disclaimers). The downloaded pages are
not redistributed in this repository; `rag/crawl.py` fetches them. Embedding model: BAAI/bge-small-en-v1.5 (MIT).
Code: MIT.
