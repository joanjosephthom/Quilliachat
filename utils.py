"""Utility functions for the RAG application."""

import os
import hashlib
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch

def compute_file_hash(file_content: bytes) -> str:
    """Compute a hash for a file to use as a unique identifier."""
    return hashlib.md5(file_content).hexdigest()

def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def count_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    This is a simple approximation. For more accurate counts,
    use the tokenizer from the specific model.
    """
    return len(text.split())

def create_cache_dir() -> str:
    """Create a cache directory if it doesn't exist."""
    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def format_sources(chunks: List[Dict[Any, Any]]) -> str:
    """Format source chunks for display."""
    sources = []
    for i, chunk in enumerate(chunks):
        source = f"Source {i+1}: {chunk['document']} (Page {chunk['page']})"
        if 'section' in chunk and chunk['section']:
            source += f", Section: {chunk['section']}"
        source += f"\n{chunk['text']}\n"
        sources.append(source)
    return "\n".join(sources)

def merge_results(bm25_results: List[Tuple[Dict, float]], 
                 dense_results: List[Tuple[Dict, float]], 
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 top_k: int = 5) -> List[Dict]:
    """Merge and rank results from BM25 and dense retrievers."""
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Process BM25 results
    for chunk, score in bm25_results:
        chunk_id = chunk.get('id', str(chunk))
        if chunk_id not in combined_scores:
            combined_scores[chunk_id] = {'chunk': chunk, 'score': 0}
        combined_scores[chunk_id]['score'] += score * bm25_weight
    
    # Process dense results
    for chunk, score in dense_results:
        chunk_id = chunk.get('id', str(chunk))
        if chunk_id not in combined_scores:
            combined_scores[chunk_id] = {'chunk': chunk, 'score': 0}
        combined_scores[chunk_id]['score'] += score * dense_weight
    
    # Sort by combined score and return top_k
    sorted_results = sorted(combined_scores.values(), 
                           key=lambda x: x['score'], 
                           reverse=True)[:top_k]
    
    return [item['chunk'] for item in sorted_results]

# --- Paraphrasing Model (T5) ---
PARAPHRASE_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
paraphrase_tokenizer = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL_NAME, use_fast=False)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL_NAME)

def generate_paraphrases(query, num_return_sequences=3, num_beams=5):
    """
    Generate paraphrases for the input query using T5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paraphrase_model.to(device)
    text =  "paraphrase: " + query + " </s>"
    encoding = paraphrase_tokenizer.encode_plus(
        text, padding='longest', return_tensors="pt"
    )
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    outputs = paraphrase_model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=64,
        do_sample=True,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams
    )
    paraphrases = set()
    for output in outputs:
        paraphrased = paraphrase_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrases.add(paraphrased)
    return list(paraphrases)

# --- Token Counting for LLM Context Limiting ---
def count_tokens_tiktoken(text, model_name="llama-3-70b"):
    """
    Count tokens in a string using tiktoken for the specified model.
    """
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback
    return len(enc.encode(text))

def build_token_limited_context(chunks, max_tokens=8000, model_name="llama-3-70b"):
    """
    Concatenate chunk texts in order, stopping before exceeding max_tokens.
    Returns the context string and the list of included chunks.
    """
    context = ""
    total_tokens = 0
    included_chunks = []
    for chunk in chunks:
        chunk_text = chunk["text"]
        chunk_tokens = count_tokens_tiktoken(chunk_text, model_name)
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += "\n\n" + chunk_text
        total_tokens += chunk_tokens
        included_chunks.append(chunk)
    return context.strip(), included_chunks
