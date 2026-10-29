"""Turn the downloaded handbook pages into retrieval chunks (data/chunks.jsonl).

Each chunk keeps its page title (e.g. "6.3.2.1. Shewhart X-bar and R and S Control Charts")
and canonical URL so every answer can cite where it came from.
Known limitation: many handbook formulas are images, so their content is not in the text.

Usage: python -m rag.chunk
"""
import html
import json
import re
from pathlib import Path

from .crawl import BASE, RAW

DATA = RAW.parent
OUT = DATA / "chunks.jsonl"
TARGET_WORDS = 180   # chunk size; small enough that a citation points at the right passage
MIN_WORDS = 25       # navigation-only pages fall below this and are dropped


def page_text(raw):
    title = re.search(r"<title>(.*?)</title>", raw, flags=re.I | re.S)
    title = re.sub(r"\s+", " ", html.unescape(title.group(1))).strip() if title else ""
    body = re.sub(r"(?s)<!--.*?-->", " ", raw)  # old image-based formulas live in comments
    body = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", body)
    body = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</tr>|</h\d>|</pre>", "\n\n", body)
    body = re.sub(r"(?i)<li[^>]*>", "\n- ", body)
    body = html.unescape(re.sub(r"<[^>]+>", " ", body))
    paras = []
    for block in re.split(r"\n\s*\n", body):
        text = re.sub(r"[ \t\r\f\v]+", " ", block).strip()
        text = re.sub(r"\n\s*", " ", text)
        # handbook chrome: chapter breadcrumbs, nav buttons, footer
        if not text or re.fullmatch(r"[\d.]*\s*", text) or text in {"Home", "Tools & Aids", "Search Handbook",
                                                                      "Previous Page", "Next Page"}:
            continue
        paras.append(text)
    return title, paras


def split_long(paras):
    """Break paragraphs far above the target size (data listings, long <pre> blocks) into word windows."""
    out = []
    for p in paras:
        words = p.split()
        if len(words) <= 2 * TARGET_WORDS:
            out.append(p)
        else:
            out += [" ".join(words[i:i + TARGET_WORDS]) for i in range(0, len(words), TARGET_WORDS)]
    return out


def is_toc(text):
    # table-of-contents chunks are lists of "[6.3.2.1.]" section numbers; they match every query
    return len(re.findall(r"\[\d+(?:\.\d+)*\.\]", text)) >= 4


def chunk_page(title, paras, url):
    chunks, cur = [], []
    for p in split_long(paras):
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur = cur[-1:] if len(cur[-1].split()) < TARGET_WORDS // 2 else []  # one-paragraph overlap
    if cur and (not chunks or sum(len(x.split()) for x in cur) >= MIN_WORDS):
        chunks.append(" ".join(cur))
    return [{"title": title, "url": url, "text": c} for c in chunks
            if len(c.split()) >= MIN_WORDS and not is_toc(c)]


def build():
    rows = []
    for f in sorted(RAW.rglob("*.htm")):
        rel = f.relative_to(RAW).as_posix()
        title, paras = page_text(f.read_text(encoding="latin-1"))
        rows += chunk_page(title, paras, BASE + rel)
    with OUT.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"id": f"c{i:04d}", **r}, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    rows = build()
    print(f"{len(rows)} chunks from {len({r['url'] for r in rows})} pages -> {OUT}")
