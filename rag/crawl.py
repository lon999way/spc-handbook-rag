"""Download chapters 5 (Process Improvement / DOE) and 6 (Process Monitoring & Control / SPC)
of the NIST/SEMATECH e-Handbook of Statistical Methods.

The handbook is public information published by NIST (https://www.nist.gov/copyrights-disclaimers);
cite it as: NIST/SEMATECH e-Handbook of Statistical Methods, https://doi.org/10.18434/M32189.

Usage: python -m rag.crawl            (polite: one request per second, skips files already saved)
"""
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

BASE = "https://www.itl.nist.gov/div898/handbook/"
CHAPTERS = {"pri": "pri/pri.htm", "pmc": "pmc/pmc.htm"}
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = "Mozilla/5.0 (portfolio RAG demo; polite crawler, 1 req/s)"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("latin-1")


def local_path(url):
    return RAW / urlparse(url).path.split("/handbook/", 1)[1]


def crawl(chapter, start):
    todo, seen = [urljoin(BASE, start)], set()
    prefix = urljoin(BASE, chapter + "/")
    while todo:
        url = todo.pop(0)
        if url in seen:
            continue
        seen.add(url)
        dest = local_path(url)
        if dest.exists():
            page = dest.read_text(encoding="latin-1")
        else:
            page = fetch(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(page, encoding="latin-1")
            time.sleep(1)
        for href in re.findall(r'href="([^"#]+\.htm)"', page, flags=re.I):
            nxt = urljoin(url, href)
            # stay inside the chapter; skip the per-chapter printable copy
            if nxt.startswith(prefix) and nxt not in seen:
                todo.append(nxt)
    return seen


if __name__ == "__main__":
    total = 0
    for ch, start in CHAPTERS.items():
        pages = crawl(ch, start)
        total += len(pages)
        print(f"{ch}: {len(pages)} pages")
    print(f"total {total} pages -> {RAW}")
