# Quillia: Turning pages into conversations...

**Quillia** is an AI-powered document assistant that lets you upload PDFs and ask questions, receiving accurate, referenced answers grounded in your documents.  
It combines state-of-the-art retrieval (hybrid dense/sparse), semantic chunking, and large language models (LLMs) for a seamless, conversational document experience.

---

## ✨ Features

- **PDF Upload & Parsing:** Upload any PDF and Quillia will extract, chunk, and index its content.
- **Modern Chat UI:** Ask questions in a clean, chat-style interface with user/AI bubbles and document summaries.
- **Hybrid Retrieval:** Combines BM25 (keyword) and dense (semantic) search for robust context retrieval.
- **Query Expansion:** Optionally paraphrase your question to improve retrieval coverage.
- **LLM-Powered Answers:** Choose between Groq Llama 3 or DeepSeek R1 (via OpenRouter) for answer generation.
- **Source Attribution:** Answers include concise, referenced source snippets (section, page, document).
- **Token-Limited Context:** Automatically manages LLM context size to avoid API errors.
- **Streamlit Cloud Ready:** Deployable on [Streamlit Cloud](https://streamlit.io/cloud) with a single click.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **PDF Parsing:** [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Retrieval:** [sentence-transformers](https://www.sbert.net/), [faiss-cpu](https://github.com/facebookresearch/faiss), [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- **Paraphrasing:** T5-based model (`Vamsi/T5_Paraphrase_Paws`)
- **LLMs:** [Groq Llama 3](https://console.groq.com/), [DeepSeek R1 via OpenRouter](https://openrouter.ai/)
- **Token Counting:** [tiktoken](https://github.com/openai/tiktoken)
- **Other:** numpy, scikit-learn, tqdm, pillow, sentencepiece

---


