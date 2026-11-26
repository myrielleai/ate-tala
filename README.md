# Ate TALA: Trusted Advocacy & Legal Assistant for Barangay VAW and GBV Safe Spaces

A retrieval-augmented (RAG) chatbot that answers questions from barangay and LGU personnel about handling
Violence Against Women (VAW) and Gender-Based Violence (GBV) cases, in English or Filipino, using only
official Philippine documents (RA 9262 and IRR, RA 11313, RA 7877, RA 9710, Barangay VAW Desk Handbook,
SAFER Women module) and citing the source page.

> Legal information only, not legal advice. In an emergency, call 911.

## How this deployment works
| Part | Runs on |
|---|---|
| Streamlit interface | Streamlit Community Cloud |
| Retrieval: multilingual-e5-small embeddings + FAISS index, built from `chunks.pkl` on first start | inside the app (CPU) |
| Answer generation | an OpenAI-compatible LLM API, set in Streamlit secrets |

The evaluated system (see the notebook) used BGE-M3 + a cross-encoder reranker and GPU-hosted models.
Because Community Cloud has no GPU and limited memory, this deployment uses the lighter e5-small retriever
without reranking and calls the LLM through an API. The prompt (`tala_core.py`) is identical.

## Deploy
1. Create a GitHub repository and upload every file in this folder (keep the folder structure).
   Do **not** upload a real `.streamlit/secrets.toml`.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose the repository,
   and set the main file to `streamlit_app.py`.
3. Open **Advanced settings**, choose Python 3.12, and paste your secrets
   (see `.streamlit/secrets.toml.example`): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
4. Deploy. The first start takes a few minutes while the embedding model downloads and the index is built.

## Files
| File | Purpose |
|---|---|
| `streamlit_app.py` | the web app |
| `tala_core.py` | retrieval and prompt (same as the evaluation notebook) |
| `chunks.pkl` | 817 cleaned, section-aware chunks from the 6 documents |
| `build_chunks.py` | rebuilds `chunks.pkl` from `docs/` if the documents change (`python build_chunks.py`) |
| `docs/` | the 6 source PDFs |
| `deploy_config.json` | embedder, top-k and not-found threshold settings |

## Run locally
```
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it
streamlit run streamlit_app.py
```
