import streamlit as st
import fitz
import time
from config import (
    RETRIEVAL_STRATEGY, TOP_K_CHUNKS, APP_TITLE, APP_DESCRIPTION,
    GROQ_API_KEY, OPENROUTER_API_KEY, EMBEDDING_MODELS
)
from indexers.index_builder import IndexBuilder
from llm.groq_llm import GroqLLM
from llm.DeepSeekR1 import DeepSeekR1
from utils import compute_file_hash
from utils import generate_paraphrases, build_token_limited_context
import numpy


# --- Accent Colors ---
ACCENT_BLUE = "#5380ff"
USER_BUBBLE_BG = "#232f4b"
INPUT_BG = "#18181a"
FOOTER_BG = "#0E1117"

# --- Custom CSS for Modern Chat and Layout ---
st.markdown(f"""
<style>
    .quillia-header {{
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }}
    .quillia-title {{
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -1px;
        color: {ACCENT_BLUE};
        margin-bottom: 0.1rem;
    }}
    .quillia-tagline {{
        font-size: 1.2rem;
        color: #b0b0b8;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }}
    .chat-area {{
        width: 100%;
        max-width: 540px;
        min-height: 60vh;
        background: rgba(36,36,40,0.92);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
        padding: 2.2rem 1.2rem 1.2rem 1.2rem;
        margin-bottom: 1.2rem;
        margin-top: 0.5rem;
        overflow-y: auto;
        scroll-behavior: smooth;
    }}
    .bubble {{
        display: flex;
        align-items: flex-end;
        margin-bottom: 0.7rem;
        max-width: 85%;
    }}
    .bubble-user {{
        justify-content: flex-end;
        margin-left: 15%;
    }}
    .bubble-content {{
        padding: 0.85rem 1.2rem;
        border-radius: 18px;
        font-size: 1.08rem;
        word-break: break-word;
        box-shadow: 0 2px 8px 0 rgba(31, 38, 135, 0.10);
        max-width: 100%;
        display: flex;
        align-items: center;
        background: {USER_BUBBLE_BG};
        color: #fff;
        text-align: right;
        flex-direction: row-reverse;
        border-bottom-right-radius: 4px;
        border-top-right-radius: 18px;
        border-top-left-radius: 18px;
        border-bottom-left-radius: 18px;
        border: 1.5px solid {ACCENT_BLUE};
    }}
    .bubble-icon {{
        width: 28px;
        height: 28px;
        margin: 0 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
    }}
    .chat-input-area {{
        width: 100%;
        max-width: 540px;
        margin: 0 auto 0.5rem auto;
        position: relative;
        z-index: 10;
        background: {INPUT_BG};
        border-radius: 12px;
        padding-bottom: 0.5rem;
    }}
    .footer-fixed {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        background: {FOOTER_BG};
        color: #888;
        font-size: 1.05rem;
        text-align: center;
        padding: 0.7rem 0 0.5rem 0;
        z-index: 100;
        border-top: 1px solid #222;
        letter-spacing: 0.01em;
    }}
    @media (max-width: 700px) {{
        .chat-area, .chat-input-area {{ max-width: 98vw; }}
        .quillia-title {{ font-size: 1.5rem; }}
        .quillia-tagline {{ font-size: 1.05rem; }}
    }}
</style>
""", unsafe_allow_html=True)

# --- SVG Icon for User ---
USER_ICON = """
<svg class="bubble-icon" viewBox="0 0 32 32" fill="none">
<circle cx="16" cy="12" r="7" fill="#5380ff"/>
<ellipse cx="16" cy="25" rx="10" ry="6" fill="#5380ff"/>
</svg>
"""

# --- Session State Initialization ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "initial"
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None
if "index_builder" not in st.session_state:
    st.session_state.index_builder = IndexBuilder()
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Groq Llama 3"
if "llm" not in st.session_state:
    st.session_state.llm = GroqLLM()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever_type" not in st.session_state:
    st.session_state.retriever_type = RETRIEVAL_STRATEGY
if "top_k_chunks" not in st.session_state:
    st.session_state.top_k_chunks = TOP_K_CHUNKS
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# --- Helper Functions ---

def select_summary_chunks(chunks, max_chunks=3):
    """
    Select the best chunks for summarization:
    - Prefer chunks with 'abstract', 'introduction', 'summary', 'overview' in section title.
    - Skip chunks with 'license', 'copyright', 'table of contents'.
    - If none found, use the first N non-empty, non-boilerplate chunks.
    """
    preferred_keywords = ["abstract", "introduction", "summary", "overview"]
    skip_keywords = ["license", "copyright", "table of contents", "permission"]

    # Prefer chunks with preferred keywords in section
    preferred = [
        chunk for chunk in chunks
        if chunk.get("section") and any(k in chunk["section"].lower() for k in preferred_keywords)
    ]
    if preferred:
        return preferred[:max_chunks]

    # Otherwise, skip chunks with skip keywords in section or text
    filtered = [
        chunk for chunk in chunks
        if not any(k in (chunk.get("section", "") + " " + chunk.get("text", "")).lower() for k in skip_keywords)
    ]
    if filtered:
        return filtered[:max_chunks]

    # Fallback: just use the first N chunks
    return chunks[:max_chunks]

def generate_llm_summary(llm, chunks, max_chunks=3, max_tokens=5000, model_name="llama-3-70b"):
    """Generate a summary using the LLM and the best available chunks, with token limit."""
    selected_chunks = select_summary_chunks(chunks, max_chunks)
    if not selected_chunks:
        return "No content to summarize."
    # Limit the context by tokens
    context, _ = build_token_limited_context(selected_chunks, max_tokens=max_tokens, model_name=model_name)
    prompt = (
        "Summarize the following document in 3-4 sentences for a user who wants to know what it's about. "
        "Be concise and clear.\n\n"
        f"Document Content:\n{context}"
    )
    summary = llm.generate_response(prompt, [], include_sources=False)
    return summary.strip()

# --- Sidebar (Settings & About) ---
with st.sidebar:
    if st.button("Home", use_container_width=True):
        st.session_state.app_mode = "initial"
        st.session_state.uploaded_file_bytes = None
        st.session_state.uploaded_file_name = None
        st.session_state.file_hash = None
        st.session_state.doc_summary = None
        st.session_state.messages = []
        st.session_state.index_builder = IndexBuilder()
        if st.session_state.get("llm_choice", "Groq Llama 3") == "Groq Llama 3":
            st.session_state.llm = GroqLLM()
        else:
            st.session_state.llm = DeepSeekR1()
        st.rerun()

    # --- Advanced Settings (in expander) ---
    with st.expander("⚙️ &nbsp; Advanced Settings", expanded=False):
        # LLM selection
        llm_options = ["Groq Llama 3", "DeepSeek R1"]
        llm_choice = st.selectbox("Language Model", llm_options, index=llm_options.index(st.session_state.llm_choice))
        if llm_choice != st.session_state.llm_choice:
            st.session_state.llm_choice = llm_choice
            if llm_choice == "Groq Llama 3":
                st.session_state.llm = GroqLLM()
            elif llm_choice == "DeepSeek R1":
                st.session_state.llm = DeepSeekR1()

        # Retriever selection
        retriever_options_map = {
            "bm25": "BM25 (Keywords)",
            "dense": "Dense Vector (Semantic)",
            "hybrid": "Hybrid (Keywords + Semantic)"
        }
        if "retriever_type" not in st.session_state:
            st.session_state.retriever_type = RETRIEVAL_STRATEGY
        current_retriever_index = list(retriever_options_map.keys()).index(st.session_state.retriever_type)
        st.session_state.retriever_type = st.selectbox(
            "Retrieval Method",
            options=list(retriever_options_map.keys()),
            format_func=lambda x: retriever_options_map[x],
            index=current_retriever_index,
            key="retriever_select"
        )
        # Embedding model selection (disabled if BM25)
        embedding_model_options = list(EMBEDDING_MODELS.keys())
        if "embedding_model_choice" not in st.session_state:
            st.session_state.embedding_model_choice = embedding_model_options[0]
        embedding_disabled = st.session_state.retriever_type == "bm25"
        selected_embedding_model = st.selectbox(
            "Sentence-Transformer Model (embedding)",
            embedding_model_options,
            index=embedding_model_options.index(st.session_state.embedding_model_choice),
            disabled=embedding_disabled,
            key="embedding_model_select"
        )
        if selected_embedding_model != st.session_state.embedding_model_choice:
            st.session_state.embedding_model_choice = selected_embedding_model
            st.session_state.index_builder = IndexBuilder()
            # Clear all document-related state to force re-upload and re-indexing
            st.session_state.uploaded_file_bytes = None
            st.session_state.uploaded_file_name = None
            st.session_state.file_hash = None
            st.session_state.doc_summary = None
            st.session_state.messages = []
            st.session_state.app_mode = "initial"  # <--- This is the key line!
            st.rerun()

            #Paraphrasing enabler
        use_query_expansion = st.checkbox("Enable Query Expansion (Paraphrasing)", value=True)

        # Chunk count slider
        st.session_state.top_k_chunks = st.slider(
            "Number of Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k_chunks,
            key="top_k_slider"
        )
            # --- About Quillia (always visible) ---
    st.markdown("""
<div class="about-section">
<strong>Quillia: Turning pages into conversations...</strong>
<br><br>
Quillia is an AI-powered document assistant that lets you ask questions and receive accurate answers grounded in your uploaded PDFs.
<br><br>
<span class="section-title">How it works (RAG pipeline):</span>
<ul>
<li>📄 <b>PDF parsing and intelligent chunking</b> structure your document into context-aware segments</li>
<li>🔍 <b>Hybrid retrieval</b> (dense + sparse using FAISS and BM25) finds the most relevant information</li>
<li>🧠 <b>Sentence Transformers</b> create semantic embeddings of document chunks</li>
<li>🤖 <b>LLaMA 3 via Groq API, Deepseek R1 via Openrouter API</b> acts as the language model, generating grounded and context-aware answers</li>
</ul>
<hr style="border: 0; border-top: 1px solid #444; margin: 1.2em 0;">
Built with ♥ by Joan Joseph Thomas
<hr style="border: 0; border-top: 1px solid #444; margin: 1.2em 0;">
Powered by Streamlit, FAISS, Groq, Openrouter and open-source RAG components
</div>
""", unsafe_allow_html=True)

# --- App Header (centered, not fixed) ---
st.markdown("""
<div class="quillia-header">
    <div class="quillia-title">Quillia</div>
    <div class="quillia-tagline">Turning pages into conversations...</div>
</div>
""", unsafe_allow_html=True)

# === SCREEN 1: Initial File Upload ===
if st.session_state.app_mode == "initial":
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 3, 2])
    uploaded_file = st.file_uploader(
        "",
        type="pdf",
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        st.session_state.app_mode = "processing"
        st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.file_hash = compute_file_hash(st.session_state.uploaded_file_bytes)
        st.session_state.messages = []
        st.session_state.index_builder = IndexBuilder()
        st.rerun()
    # Footer (fixed)
    st.markdown(f'<div class="footer-fixed">Quillia &copy; 2025 &mdash; Built with ♥ by Joan Joseph Thomas</div>', unsafe_allow_html=True)

# === SCREEN 1.5: Processing Uploaded File ===
if st.session_state.app_mode == "processing":
    with st.spinner(f"Processing {st.session_state.uploaded_file_name}..."):
        if st.session_state.uploaded_file_bytes is None:
             st.error("File data is missing. Please re-upload.")
             st.session_state.app_mode = "initial"
             st.rerun()
        try:
            chunks = st.session_state.index_builder.process_file_bytes(
                st.session_state.uploaded_file_bytes, st.session_state.uploaded_file_name
            )
            if not chunks:
                st.warning("Could not extract any content chunks from the PDF.")
        except Exception as e:
            st.error(f"Error processing PDF for RAG: {e}")
            st.session_state.app_mode = "initial"
            st.session_state.uploaded_file_bytes = None
            time.sleep(2)
            st.rerun()
        # Generate a true summary using the LLM and best chunks
        st.session_state.doc_summary = generate_llm_summary(st.session_state.llm, chunks)
        # Add summary as the first message in chat history
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"**Document Summary:**\n\n{st.session_state.doc_summary}"
        }]
        st.session_state.app_mode = "chat"
        st.rerun()

# === SCREEN 2: Chat Only (Modern, Centered, User Bubble, Fixed Footer) ===
if st.session_state.app_mode == "chat":
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="bubble bubble-user"><span class="bubble-content">{message["content"]}{USER_ICON}</span></div><div style="clear:both;"></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(message["content"])
    # Auto-scroll to bottom JS
    st.markdown("""
    <script>
    var chatArea = window.parent.document.getElementById("chat-area");
    if (chatArea) { chatArea.scrollTop = chatArea.scrollHeight; }
    </script>
    """, unsafe_allow_html=True)
    # Chat input at bottom, above fixed footer
    st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
    if not st.session_state.is_generating:
        prompt = st.chat_input(
            f"Ask about {st.session_state.uploaded_file_name if st.session_state.uploaded_file_name else 'your document'}...",
            key="chat_input_main",
            disabled=(st.session_state.app_mode != "chat")
        )
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.is_generating = True
            st.rerun()
    else:
        st.chat_input(
            f"Generating answer...",
            key="chat_input_disabled",
            disabled=True
        )
        st.info("Generating answer...")
    st.markdown('</div>', unsafe_allow_html=True)
    # Fixed Footer
    st.markdown(f'<div class="footer-fixed">Quillia &copy; 2025 &mdash; Built with ♥ by Joan Joseph Thomas</div>', unsafe_allow_html=True)

# --- Handle Response Generation AFTER potential rerun from input ---
if (
    st.session_state.app_mode == "chat"
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and st.session_state.is_generating
):
    with st.spinner("Generating answer..."):
        full_response = ""
        try:
            if "index_builder" not in st.session_state or st.session_state.index_builder is None:
                raise ValueError("Index builder not initialized.")
            retriever = st.session_state.index_builder.get_retriever(st.session_state.retriever_type)
            if retriever is None:
                raise ValueError(f"Could not find retriever: {st.session_state.retriever_type}")
            last_user_prompt = st.session_state.messages[-1]["content"]

            # --- Query Expansion ---
            if use_query_expansion:
                paraphrases = generate_paraphrases(last_user_prompt, num_return_sequences=3)
                all_queries = [last_user_prompt] + paraphrases
            else:
                all_queries = [last_user_prompt]

            # --- Retrieve for each query ---
            all_chunks = []
            for q in all_queries:
                if st.session_state.retriever_type == "hybrid":
                    chunks = retriever.retrieve(q, top_k=st.session_state.top_k_chunks)
                else:
                    chunk_score_pairs = retriever.retrieve(q, top_k=st.session_state.top_k_chunks)
                    chunks = [chunk for chunk, _ in chunk_score_pairs]
                all_chunks.extend(chunks)

            # --- Deduplicate chunks (by id) ---
            unique_chunks = {chunk['id']: chunk for chunk in all_chunks}.values()

            # --- Use the top N by retriever order ---
            retrieved_chunks = list(unique_chunks)[:st.session_state.top_k_chunks]

            if not retrieved_chunks:
                st.warning("No relevant document chunks found to answer the question.")

            if "llm" not in st.session_state or st.session_state.llm is None:
                raise ValueError("LLM not initialized.")

            llm_response = st.session_state.llm.generate_response(
                last_user_prompt, retrieved_chunks
            )
            words = llm_response.split()
            if not words:
                full_response = "I received an empty response."
            else:
                for i in range(len(words)):
                    full_response = " ".join(words[:i+1])
                    time.sleep(0.02)
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
        except Exception as e:
            full_response = f"Sorry, an error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.is_generating = False
        st.rerun()
