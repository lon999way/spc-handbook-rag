"""Streamlit front end. Run: streamlit run app.py"""
import streamlit as st

from rag.answer import MIN_COSINE, ask

st.set_page_config(page_title="SPC Handbook Assistant", page_icon="📈")
st.title("SPC & DOE Handbook Assistant")
st.caption("Answers only from the NIST/SEMATECH e-Handbook of Statistical Methods (chapters 5–6), "
           "with citations. Capability numbers are computed by code, not by the model.")

examples = [
    "What is the difference between Cp and Cpk?",
    "When should I use an EWMA chart instead of a Shewhart chart?",
    "Compute Cpk for 10.1, 9.9, 10.3, 10.0, 10.2, 9.8 with spec limits 9.5 to 10.7",
    "What was Micron's HBM yield last quarter?",
]
q = st.text_input("Ask a question", placeholder=examples[0])
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex[:28] + "…", help=ex):
        q = ex

if q:
    with st.spinner("Searching the handbook…"):
        try:
            out = ask(q)
        except RuntimeError as e:
            st.error(f"Model backend unavailable: {e}")
            st.stop()
    if out["refused"]:
        st.warning(out["answer"])
    else:
        st.markdown(out["answer"])
    if out["tool_result"]:
        st.subheader("Tool result (process_capability)")
        st.json(out["tool_result"])
    if out["citations"]:
        st.subheader("Sources")
        for cid in out["citations"]:
            hit = next(h for h in out["hits"] if h["id"] == cid)
            st.markdown(f"- [{hit['title']}]({hit['url']})")
    with st.expander("Retrieval trace"):
        st.write(f"Refusal threshold: best cosine < {MIN_COSINE}; model calls: {out['llm_calls']}; "
                 f"gate: {out['gate']}")
        st.table([{k: h[k] for k in ("id", "cosine", "title")} for h in out["hits"]])
