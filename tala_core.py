"""Ate TALA core: retrieval, prompting and generation (shared by the notebook and the Streamlit app)."""
import re, pickle, threading, time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

EMBEDDERS = {
    "e5-small": {"id": "intfloat/multilingual-e5-small", "q": "query: ", "p": "passage: "},
    "e5-large": {"id": "intfloat/multilingual-e5-large", "q": "query: ", "p": "passage: "},
    "bge-m3":   {"id": "BAAI/bge-m3", "q": "", "p": ""},
}
RERANKER_ID = "BAAI/bge-reranker-v2-m3"

NOT_FOUND_FIL = "Paumanhin, wala ito sa mga dokumentong hawak ko."
NOT_FOUND_EN = "I could not find this in my documents."
NOT_FOUND_MSG = NOT_FOUND_FIL + " / " + NOT_FOUND_EN

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are Ate TALA (Trusted Advocacy & Legal Assistant), a reference assistant for Philippine barangay and LGU personnel (VAW Desk officers, Punong Barangay, kagawad, social workers, GAD focal persons) who handle Violence Against Women (VAW) and Gender-Based Violence (GBV) cases.

Rules:
1. Use ONLY the information in the SOURCES. Do not add facts, agencies, laws, numbers or penalties that are not in the SOURCES.
2. If the SOURCES answer the question fully or partly, answer with what they say, including for scenario questions. Only if NONE of the SOURCES relate to the question, reply exactly: "__NF__" Then, in one sentence, suggest who to consult (MSWDO/CSWDO, PNP Women and Children Protection Desk, PAO, or PCW), and stop.
3. After each key point, cite its source using the short label exactly as written in the SOURCE line, for example [RA9262-IRR, p.12].
4. Be concise: at most 150 words. Use numbered steps for procedures. Do not repeat or summarize the sources one by one.
5. Be survivor-centered: keep information confidential, never blame the victim, and never recommend mediation or amicable settlement of VAW cases.
6. If someone may be in immediate danger, first advise calling 911 or going to the nearest police station or barangay VAW Desk.
7. Give legal information, not legal advice.
8. LANGUAGE: If the QUESTION is in English, answer in English. If the QUESTION is in Filipino or Taglish, answer in Filipino. Always write numbers as digits exactly as in the SOURCES (for example "15 days" / "15 araw", "P10,000", "5%"), and keep legal terms such as BPO, TPO, PPO and VAWC in English.""".replace("__NF__", NOT_FOUND_MSG)


def page_label(pages):
    pages = sorted(set(pages))
    return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"


def source_label(hit):
    return f"[{hit.get('label') or hit['source']}, {page_label(hit['pages'])}]"


class Retriever:
    """Embeds the question, searches the FAISS index, optionally reranks."""

    def __init__(self, embedder_key, index, chunks, use_reranker=False, embedder=None, reranker=None):
        self.key = embedder_key
        self.spec = EMBEDDERS[embedder_key]
        self.index = index
        self.chunks = chunks
        self.embedder = embedder or SentenceTransformer(self.spec["id"])
        self.reranker = None
        if use_reranker:
            self.reranker = reranker or CrossEncoder(RERANKER_ID, max_length=512)

    @classmethod
    def from_files(cls, index_path, chunks_path, embedder_key, use_reranker=False):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return cls(embedder_key, index, chunks, use_reranker)

    def search(self, query, k=5, fetch_k=20):
        q = self.embedder.encode([self.spec["q"] + query], normalize_embeddings=True).astype("float32")
        n = fetch_k if self.reranker is not None else k
        scores, ids = self.index.search(q, n)
        hits = [dict(self.chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
        if self.reranker is not None and hits:
            rs = self.reranker.predict([(query, h["text"]) for h in hits])
            for h, r in zip(hits, rs):
                h["rerank_score"] = float(r)
            hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        hits = hits[:k]
        for rank, h in enumerate(hits, 1):
            h["rank"] = rank
        return hits

    @staticmethod
    def top_score(hits):
        if not hits:
            return float("-inf")
        return hits[0].get("rerank_score", hits[0]["score"])


def build_messages(question, hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        sec = f" | {h['section']}" if h.get("section") else ""
        blocks.append(f"SOURCE {i}: {source_label(h)}{sec}\n{h['text']}")
    user = "SOURCES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class Generator:
    """Loads one Hugging Face chat model and generates answers (plain or streamed)."""

    def __init__(self, model_id, max_new_tokens=600):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=pick_dtype(), device_map="auto")
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def prompt_text(self, messages):
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            # Some templates (e.g. Gemma) have no system role: merge it into the user turn.
            merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
            return self.tokenizer.apply_chat_template(merged, **kwargs)

    def _inputs(self, messages):
        text = self.prompt_text(messages)
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

    def _gen_kwargs(self):
        return dict(max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id)

    @staticmethod
    def clean(text):
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    @torch.no_grad()
    def generate(self, messages):
        inputs = self._inputs(messages)
        start = time.time()
        out = self.model.generate(**inputs, **self._gen_kwargs())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.time() - start
        new = out[0][inputs["input_ids"].shape[-1]:]
        text = self.clean(self.tokenizer.decode(new, skip_special_tokens=True))
        return text, int(new.shape[-1]), seconds, int(inputs["input_ids"].shape[-1])

    def stream(self, messages):
        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(target=self.model.generate, kwargs=dict(**inputs, **self._gen_kwargs(), streamer=streamer))
        thread.start()
        for piece in streamer:
            yield piece
        thread.join()


def answer(question, retriever, generator, k=5, threshold=None):
    """Full RAG answer: retrieve, (optionally) refuse on low score, generate."""
    hits = retriever.search(question, k=k)
    if threshold is not None and Retriever.top_score(hits) < threshold:
        return {"answer": NOT_FOUND_MSG, "hits": hits, "seconds": 0.0, "new_tokens": 0,
                "prompt_tokens": 0, "refused_by_threshold": True}
    text, ntok, secs, ptok = generator.generate(build_messages(question, hits))
    return {"answer": text, "hits": hits, "seconds": secs, "new_tokens": ntok,
            "prompt_tokens": ptok, "refused_by_threshold": False}
