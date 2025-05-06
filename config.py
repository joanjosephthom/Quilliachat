import os

# Try to import st.secrets if running in Streamlit, else fallback
try:
    import streamlit as st
    _st_secrets = st.secrets
except (ImportError, AttributeError):
    _st_secrets = {}

def get_secret(key, default=""):
    # Priority: st.secrets > environment variable > default
    return _st_secrets.get(key, os.environ.get(key, default))

# Retrieval settings
RETRIEVAL_STRATEGY = "hybrid"
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7
TOP_K_CHUNKS = 5

# Chunking settings
MAX_CHUNK_SIZE = 500
MIN_CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Model settings
EMBEDDING_MODELS = {
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
}
EMBEDDING_MODEL = EMBEDDING_MODELS["all-MiniLM-L6-v2"]
LLM_MODEL = "llama-3.3-70b-versatile"
DEEPSEEK_MODEL = "deepseek/deepseek-r1"

# API keys (secure for both local and Streamlit Cloud)
GROQ_API_KEY = get_secret("GROQ_API_KEY")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")

# UI settings
APP_TITLE = "Quillia"
APP_DESCRIPTION = "Ask questions about your PDF documents"
MAX_UPLOAD_SIZE = 200
