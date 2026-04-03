"""
Run once to pre-download models into ./models/
Used by Streamlit Cloud on first boot via @st.cache_resource
Models are auto-downloaded by HuggingFace if local path doesn't exist.
"""
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Downloading embedding model...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="./models/all-MiniLM-L6-v2")

print("Downloading LLM...")
AutoTokenizer.from_pretrained("google/flan-t5-base")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

print("Done.")
