"""Builds chunks.pkl from the PDFs in docs/ (same cleaning and chunking as the Ate TALA notebook).
Run once: python build_chunks.py   (only needed if you change the documents)"""
import os, re, pickle
from collections import Counter
from pypdf import PdfReader

DOC_DIR = "docs"
CHUNK_WORDS = 300      # words per chunk
OVERLAP = 60           # words shared between neighbouring chunks
MIN_SECTION_WORDS = 40 # very short sections are merged into the next one

HEADING_RE = re.compile(
    r"^(?:(?:MODULE|Module)\s+\d+|(?:SESSION|Session)\s+\d+"
    r"|(?:SECTION|Section|SEC\.|Sec\.)\s*\d+[A-Za-z]?\s*[.:\u2013\u2014-](?!\w)"   # "Section 5." / "SEC. 30 -", not "Section 7 hereof"
    r"|(?:RULE|Rule|ARTICLE|Article|CHAPTER|Chapter)\s+[IVXLC]+\b)"              # roman numerals only, not "Article 335"
)

def clean_text(t):
    t = t.replace("­", "")                 # soft hyphens
    t = re.sub(r"-\n(\w)", r"\1", t)           # re-join words split across lines
    t = re.sub(r"[ \t]+", " ", t)
    return t

def read_pdf(path):
    # Returns a list of pages; each page is a list of non-empty lines.
    reader = PdfReader(path)
    return [[ln.strip() for ln in clean_text(p.extract_text() or "").split("\n") if ln.strip()] for p in reader.pages]

def remove_headers_footers(pages, min_pages=5, ratio=0.4):
    # Drops short lines that repeat on many pages (running headers/footers) and bare page numbers.
    if len(pages) < min_pages:
        return pages, set()
    key = lambda ln: re.sub(r"\d+", "#", ln)          # "Page 12 of 80" and "Page 13 of 80" count as the same line
    counts = Counter(key(ln) for pg in pages for ln in set(pg) if len(ln) < 80 and not HEADING_RE.match(ln))
    repeated = {k for k, c in counts.items() if c >= ratio * len(pages)}
    cleaned = [[ln for ln in pg if (HEADING_RE.match(ln) or key(ln) not in repeated)
                and not re.fullmatch(r"(page\s*)?\d{1,4}", ln, re.I)] for pg in pages]
    return cleaned, repeated

docs = {}
for fname in sorted(f for f in os.listdir(DOC_DIR) if f.lower().endswith(".pdf")):
    pages, _ = remove_headers_footers(read_pdf(os.path.join(DOC_DIR, fname)))
    docs[fname] = pages

def split_sections(pages):
    # Returns sections as dicts: title, words, word_pages (page number of every word).
    sections, cur = [], {"title": "", "words": [], "word_pages": []}
    for pno, lines in enumerate(pages, 1):
        for ln in lines:
            if HEADING_RE.match(ln) and cur["words"]:
                sections.append(cur)
                cur = {"title": ln[:120], "words": [], "word_pages": []}
            elif HEADING_RE.match(ln):
                cur["title"] = ln[:120]
            w = ln.split()
            cur["words"] += w
            cur["word_pages"] += [pno] * len(w)
    if cur["words"]:
        sections.append(cur)
    merged, buf = [], None          # merge very short sections into the next one
    for s in sections:
        if buf:
            s = {"title": buf["title"] or s["title"], "words": buf["words"] + s["words"],
                 "word_pages": buf["word_pages"] + s["word_pages"]}
            buf = None
        if len(s["words"]) < MIN_SECTION_WORDS:
            buf = s
        else:
            merged.append(s)
    if buf:
        merged.append(buf)
    return merged

def chunk_document(fname, pages):
    out = []
    for sec in split_sections(pages):
        words, wp = sec["words"], sec["word_pages"]
        start = 0
        while start < len(words):
            end = min(start + CHUNK_WORDS, len(words))
            out.append({"source": fname, "section": sec["title"], "pages": sorted(set(wp[start:end])),
                        "text": " ".join(words[start:end]), "n_words": end - start})
            if end == len(words):
                break
            start += CHUNK_WORDS - OVERLAP
    return out

# Short labels used in citations (the full file name stays in "source" for evaluation)
SHORT_NAMES = {"9262": "RA9262-IRR", "11313": "RA11313-SafeSpaces", "7877": "RA7877-ASH",
               "9710": "RA9710-MCW", "handbook": "VAW-Desk-Handbook", "safer": "SAFER-Module"}
def short_label(fname):
    return next((v for k, v in SHORT_NAMES.items() if k in fname.lower()), os.path.splitext(fname)[0])

chunks, seen = [], set()
for fname, pages in docs.items():
    for c in chunk_document(fname, pages):
        c["label"] = short_label(fname)
        if c["text"] not in seen:
            seen.add(c["text"])
            c["id"] = len(chunks)
            chunks.append(c)


with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
counts = Counter(c["label"] for c in chunks)
print("Total chunks:", len(chunks), dict(counts))
