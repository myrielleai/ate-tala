"""Ate TALA - Streamlit Community Cloud version.

Retrieval (multilingual-e5-small + FAISS) runs inside the app on CPU; the index is built from chunks.pkl on first start.
The language model is called through an OpenAI-compatible API (Together.ai, Hugging Face,
OpenRouter, Groq, etc.), configured in Streamlit secrets. The prompt is the same one used
in the evaluation notebook (tala_core.py).
"""
import json, os, pickle
import faiss
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tala_core import Retriever, EMBEDDERS, build_messages, source_label, NOT_FOUND_MSG

CFG = json.load(open("deploy_config.json"))

st.set_page_config(page_title="Ate TALA", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading Ate TALA's knowledge base...")
def load_retriever():
    # Builds the FAISS index from chunks.pkl the first time the app starts (about 1-3 minutes on CPU),
    # then reuses it. No Colab needed.
    with open(CFG["chunks_path"], "rb") as f:
        chunks = pickle.load(f)
    spec = EMBEDDERS[CFG["embedder"]]
    model = SentenceTransformer(spec["id"])
    model.max_seq_length = 512
    if os.path.exists(CFG["index_path"]):
        index = faiss.read_index(CFG["index_path"])
    else:
        vecs = model.encode([spec["p"] + c["text"] for c in chunks], batch_size=32,
                            normalize_embeddings=True).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        try:
            faiss.write_index(index, CFG["index_path"])
        except OSError:
            pass
    return Retriever(CFG["embedder"], index, chunks, use_reranker=False, embedder=model)


@st.cache_resource
def load_client():
    return OpenAI(base_url=st.secrets["LLM_BASE_URL"], api_key=st.secrets["LLM_API_KEY"])


def stream_answer(messages):
    stream = load_client().chat.completions.create(
        model=st.secrets["LLM_MODEL"], messages=messages, stream=True,
        temperature=0, max_tokens=600,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


try:
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if k not in st.secrets]
except Exception:
    missing = ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"]
if missing:
    st.error("Missing secrets: " + ", ".join(missing) + ". Add them in your app's Settings > Secrets.")
    st.stop()

retriever = load_retriever()

with st.sidebar:
    st.header("Ate TALA")
    st.write("Trusted Advocacy & Legal Assistant for Barangay VAW and GBV Safe Spaces.")
    st.warning("Legal information only, not legal advice. In an emergency, call **911** or go to the "
               "nearest police station or barangay VAW Desk.")
    show_sources = st.toggle("Show retrieved sources", value=True)
    top_k = st.slider("Sources to retrieve (top-k)", 3, 8, CFG.get("top_k", 5))
    st.caption(f"Model: {st.secrets.get('LLM_MODEL', '?')}  \nRetrieval: {CFG['embedder']} + FAISS")
    st.caption("Knowledge base: RA 9262, RA 11313, RA 7877, RA 9710, Barangay VAW Desk Handbook, SAFER Women module.")
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("Ate TALA")
st.caption("Ask about handling VAW and GBV cases at the barangay level, in English or Filipino.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources") and show_sources:
            with st.expander("Retrieved sources"):
                st.markdown(m["sources"])

question = st.chat_input("Magtanong / Ask a question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = retriever.search(question, k=top_k)
        src_md = "\n\n".join(
            f"**{i}. {source_label(h)}** {h.get('section', '')}  \nscore: {h['score']:.3f}\n\n> {h['text'][:500]}..."
            for i, h in enumerate(hits, 1)
        )
        if show_sources:
            with st.expander("Retrieved sources"):
                st.markdown(src_md)
        threshold = CFG.get("threshold")
        if threshold is not None and Retriever.top_score(hits) < threshold:
            reply = NOT_FOUND_MSG
            st.markdown(reply)
        else:
            try:
                reply = st.write_stream(stream_answer(build_messages(question, hits)))
            except Exception as e:
                reply = "Sorry, the language model service is not responding right now. Please try again."
                st.error(f"{reply}\n\n`{type(e).__name__}`")
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": src_md})
