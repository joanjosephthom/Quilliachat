# Quilliachat: Turning Pages into Conversations

**Quilliachat** is a modular, AI-powered document assistant that allows you to upload PDFs and ask natural language questions, receiving accurate, referenced answers grounded in the document itself.

It integrates **state-of-the-art hybrid retrieval**, **semantic chunking**, and **large language models (LLMs)** into an interactive Streamlit interface. The system has been systematically evaluated over both a custom-labeled dataset and the BEIR benchmark to identify the best-performing configuration.

---

## ✨ Features

- **📄 PDF Upload & Parsing**  
  Upload any PDF. Quilliachat will extract, chunk, and index it for question answering.

- **🧠 Hybrid Retrieval Engine**  
  Combines BM25 (keyword-based) and dense (semantic) retrieval using [FAISS](https://github.com/facebookresearch/faiss) and [SentenceTransformers](https://www.sbert.net/).

- **📚 Semantic Chunking**  
  Section-aware chunking strategy with overlap and metadata (page, section, ID) to improve context retrieval.

- **📝 Query Expansion (Optional)**  
  Use a T5 paraphraser to rephrase your queries and potentially improve recall (though not enabled by default).

- **💬 Chat Interface**  
  Clean, chat-style UI built with Streamlit, showing both user and AI messages alongside source references.

- **🔍 Configurable Retrieval Settings**  
  Choose between retrievers (BM25, Dense, Hybrid), dense models (MiniLM, MPNet), top-k, and query expansion.

- **📈 Evaluated & Tuned**  
  Default settings are chosen based on extensive evaluation across accuracy, latency, and memory usage.

- **🚀 Streamlit Cloud Ready**  
  Easily deploy on [Streamlit Cloud](https://streamlit.io/cloud) with one click.

---

## 🧪 Evaluation

Quilliachat was evaluated using two datasets:

### 🔹 Custom Labeled Dataset

- 5 diverse PDFs (legal, academic, scientific, research)
- 25 manually labeled questions
- 36 configurations tested via grid search
- **Best Config (Custom):** `Hybrid + MiniLM-L6-v2 + top_k=10`
- Final default: `top_k=5` for speed/accuracy balance
- Query expansion increased resource usage with minimal gain

### 🔹 BEIR Benchmark (nfcorpus)

- 3,600+ biomedical passages, 324 queries with qrels
- Evaluated BM25, Dense (MiniLM, MPNet), and Hybrid retrievals
- **Best Config (BEIR):** `Dense + MiniLM-L6-v2 + top_k=5 + no query expansion`
- Dense & Hybrid outperform BM25 for semantic queries
- Query expansion again showed no accuracy gain but higher latency/memory

### 📊 Metrics Evaluated

- **Accuracy**, **MRR**, **Recall@k**
- **Latency**, **Memory Usage**
- **Cosine Similarity**  
  *(Full results available in the evaluation folder)*

---

## 🛠 Tech Stack

| Layer         | Tools / Libraries |
|---------------|------------------|
| Frontend      | [Streamlit](https://streamlit.io/) |
| PDF Parsing   | [PyMuPDF](https://pymupdf.readthedocs.io/), pdfplumber, pdfminer.six |
| Retrieval     | [FAISS](https://github.com/facebookresearch/faiss), [rank-bm25](https://github.com/dorianbrown/rank_bm25), [SentenceTransformers](https://www.sbert.net/) |
| Paraphrasing  | T5 (`Vamsi/T5_Paraphrase_Paws`) |
| LLMs          | [Groq LLaMA 3](https://console.groq.com/), [DeepSeek R1](https://openrouter.ai/) |
| Utilities     | `tiktoken`, `numpy`, `scikit-learn`, `tqdm`, `pillow` |

---

## 📂 Project Structure

```bash
├── chunking/           # PDF text extraction and chunking logic
├── retrievers/         # BM25, Dense, Hybrid retriever classes
├── indexers/           # Indexing logic
├── llm/                # LLM query interface
├── evaluation/         # Scripts for benchmarking & result analysis
├── utils.py            # Helper functions
├── main.py             # Streamlit app entry point

---
## 📜 References

1. [Lewis et al. (2020) - Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)  
2. [Thakur et al. (2021) - BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)  
3. [Reimers & Gurevych (2019) - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)  
4. [Ma et al. (2021) - A Replicable Analysis of Hybrid Search](https://arxiv.org/abs/2106.00882)  
5. [Nogueira et al. (2019) - Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375)  
6. [BEIR GitHub Repository](https://github.com/beir-cellar/beir)  
7. [Sentence Transformers GitHub](https://github.com/UKPLab/sentence-transformers)  

