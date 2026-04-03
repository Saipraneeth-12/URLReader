import os
import pickle
import logging

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RockyBot Offline News Research", layout="wide")
st.title("📰 RockyBot: Fully Offline News Research Tool")
st.sidebar.title("🔗 News Article URLs")

VECTORSTORE_FILE = "faiss_store_open.pkl"
EMBED_MODEL_PATH = "./models/all-MiniLM-L6-v2"
_LOCAL_LLM_PATH = "./models/flan-t5-base"
LLM_MODEL_NAME = _LOCAL_LLM_PATH if os.path.isdir(_LOCAL_LLM_PATH) else "google/flan-t5-base"
CHUNK_SIZE = 400
DEFAULT_TOP_K = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@st.cache_data(show_spinner=False)
def fetch_page_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


@st.cache_data(show_spinner=False)
def load_documents_from_urls(urls: tuple) -> list[Document]:
    docs = []
    for url in urls:
        try:
            raw_text = fetch_page_text(url)
            if raw_text:
                docs.append(Document(page_content=raw_text, metadata={"source": url}))
            else:
                st.warning(f"⚠️ Empty text from: {url}")
        except Exception as exc:
            logger.exception("Unable to fetch URL %s", url)
            st.warning(f"⚠️ Skipping {url}: {exc}")
    return docs


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )


@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=512,
        num_beams=1,
        do_sample=False,
        truncation=True,
        clean_up_tokenization_spaces=True,
    )


def build_and_save_vectorstore(chunks: list[Document]) -> FAISS:
    embeddings = load_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    with open(VECTORSTORE_FILE, "wb") as fp:
        pickle.dump(vectorstore, fp)
    return vectorstore


def load_vectorstore() -> FAISS:
    with open(VECTORSTORE_FILE, "rb") as fp:
        return pickle.load(fp)


def process_urls(urls: list[str]) -> FAISS | None:
    if not urls:
        st.error("Please enter at least one URL in the sidebar.")
        return None

    documents = load_documents_from_urls(tuple(urls))
    if not documents:
        st.error("No valid documents were loaded from the provided URLs.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        st.error("Text splitting produced zero chunks. Try different URLs.")
        return None

    vectorstore = build_and_save_vectorstore(chunks)
    st.success(f"✅ Indexed {len(chunks)} chunks from {len(documents)} article(s).")
    return vectorstore


def extract_relevant_context(query: str, docs: list[Document], top_n: int = 10) -> str:
    embedder = load_embedding_model()
    sentences = []
    for doc in docs:
        for sent in doc.page_content.split("."):
            sent = sent.strip()
            if len(sent) > 30:
                sentences.append(sent)

    if not sentences:
        return ""

    query_vec = np.array(embedder.embed_query(query))
    sent_vecs = np.array(embedder.embed_documents(sentences))
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    sent_norms = sent_vecs / (np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-9)
    scores = sent_norms @ query_norm

    top_idx = sorted(np.argsort(scores)[::-1][:top_n])
    return ". ".join(sentences[i] for i in top_idx) + "."


def synthesize_answer(query: str, context: str) -> str:
    llm = load_llm_pipeline()
    prompt = (
        f"You are a helpful assistant. Read the context carefully and write a detailed, "
        f"well-structured paragraph answering the question in your own words. "
        f"Cover all key points from the context.\n\n"
        f"Context: {context[:1800]}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return llm(prompt)[0]["generated_text"].strip()


def answer_query(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    if not query.strip():
        st.warning("Please type a question.")
        return

    if not os.path.exists(VECTORSTORE_FILE):
        st.error("No FAISS index found. Process URLs first.")
        return

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    source_docs = retriever.invoke(query)

    if not source_docs:
        st.warning("No relevant chunks found for your question.")
        return

    context = extract_relevant_context(query, source_docs, top_n=10)
    if not context.strip():
        st.warning("Could not extract relevant context. Try rephrasing.")
        return

    answer = synthesize_answer(query, context)

    st.markdown("### 📝 Answer")
    st.markdown(
        f"<div style='background:#f0f4ff;padding:16px;border-left:4px solid #4a90e2;"
        f"border-radius:6px;font-size:15px;line-height:1.8'>{answer}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🔗 Sources")
    seen = set()
    for idx, doc in enumerate(source_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        if source not in seen:
            seen.add(source)
            st.write(f"{idx}. {source}")


def main():
    st.sidebar.markdown("---")
    st.sidebar.write("### Step 1: Add source URL(s)")

    sidebar_urls = []
    for i in range(5):
        url = st.sidebar.text_input(f"URL {i + 1}", key=f"url{i}")
        if url and url.strip():
            sidebar_urls.append(url.strip())

    if st.sidebar.button("📡 Process URLs"):
        with st.spinner("Fetching and indexing pages..."):
            process_urls(sidebar_urls)

    st.sidebar.markdown("---")
    top_k = st.sidebar.slider("Top-k results", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    user_query = st.text_input("💬 Ask your question:", key="user_query")
    if st.button("🧠 Get answer"):
        with st.spinner("Running retrieval and generating answer..."):
            answer_query(user_query, top_k=top_k)

    st.sidebar.markdown("---")
    st.sidebar.caption("RockyBot — offline news research powered by LangChain.")


if __name__ == "__main__":
    main()
