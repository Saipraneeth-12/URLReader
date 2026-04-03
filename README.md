# RockyBot — Offline News Research (URL Reader)

🚀 **Live App:** [https://rockybot-news.streamlit.app](https://rockybot-news.streamlit.app)

A small Streamlit application for scraping web articles from URLs, embedding text with a local Hugging Face model, storing in FAISS, and answering user queries via LangChain retrieval QA.

---

## ✅ What this app does

- Accepts up to 5 article URLs from the sidebar
- Fetches HTML and extracts clean text using BeautifulSoup
- Splits content into chunks (RecursiveCharacterTextSplitter)
- Builds FAISS vector store with embeddings (all-MiniLM-L6-v2)
- Saves vector store locally (`faiss_store_open.pkl`)
- Runs retrieval-based `RetrievalQA` with offline T5 model
- Handles errors, provides progress and source links

---

## 📦 Files in repository

- `main.py` — streamlit app
- `README.md` — this file
- `requirements.txt` — Python dependencies
- `models/t5-small` — local T5 weights
- `models/all-MiniLM-L6-v2` — local embedding model

---

## 🛠️ Setup

1. Clone repository
2. Create/activate virtual env

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install deps

```bash
pip install -r requirements.txt
```

4. Start Streamlit app

```bash
streamlit run main.py
```

---

## 🧠 Usage

1. Open app in browser (Streamlit address shown in terminal).
2. Enter 1+ URLs in sidebar.
3. Click `📡 Process URLs`.
4. Wait until indexing is done.
5. Enter a question.
6. Click `🧠 Get answer`.
7. View answer + source list.

---

## ⚙️ Configuration keys

- `VECTORSTORE_FILE` -> saved vector index filename
- `DEFAULT_TEXT_CHUNK` -> chunk size for splitting
- `model_path` -> `./models/t5-small` (LLM)
- `embed_model_path` -> `./models/all-MiniLM-L6-v2` (embedding)

---

## 🧹 Troubleshooting

- `FileNotFoundError` for model: make sure `models/t5-small` and `models/all-MiniLM-L6-v2` exist.
- `ConnectionError` for URL fetching: check network or URL availability.
- `FAISS index not found` at query: click `Process URLs` first.

---

## 🧾 Requirements

Python 3.10+ and the following primary packages:

- streamlit
- requests
- beautifulsoup4
- transformers
- langchain
- langchain-community
- faiss-cpu

(Prefer installing via `pip install -r requirements.txt`)

---

## 🛡️ Notes

- All models are loaded from local directories for offline operation.
- Caching is used to avoid duplicate downloads and repeated model init.
- The app is designed for research and demo purposes, not production.
