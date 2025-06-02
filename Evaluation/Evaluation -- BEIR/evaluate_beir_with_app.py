import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import psutil

from indexers.index_builder import IndexBuilder
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
from utils import generate_paraphrases
from config import EMBEDDING_MODELS

RETRIEVER_OPTIONS = ["bm25", "dense", "hybrid"]
DENSE_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
QUERY_EXPANSION_OPTIONS = [False, True]
TOP_K_OPTIONS = [3, 5, 10]
RECALL_K = 10

def cosine_sim(text1, text2, model):
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beir_dir", type=str, default="datasets/nfcorpus", help="Path to BEIR nfcorpus directory")
    parser.add_argument("--output", type=str, default="beir_eval_results.csv", help="Output CSV for per-question results")
    parser.add_argument("--summary", type=str, default="beir_eval_summary.csv", help="Output CSV for per-config summary")
    parser.add_argument("--max_questions", type=int, help="Max number of questions to run (for quick tests)")
    args = parser.parse_args()

    # --- Load BEIR data ---
    with open(os.path.join(args.beir_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        corpus = {doc["_id"]: doc for doc in (json.loads(line) for line in f)}
    with open(os.path.join(args.beir_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        queries = {q["_id"]: q["text"] for q in (json.loads(line) for line in f)}
    with open(os.path.join(args.beir_dir, "qrels/test.tsv"), "r", encoding="utf-8") as f:
        qrels = {}
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) == 4:
                qid, _, docid, rel = parts
            else:
                raise ValueError(f"Unexpected qrels line format: {line}")
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(docid)

    # Only evaluate queries that have qrels
    query_ids = [qid for qid in queries if qid in qrels]
    if args.max_questions:
        query_ids = query_ids[:args.max_questions]

    print(f"Loaded {len(corpus)} documents in corpus.")
    print(f"Loaded {len(queries)} queries.")
    print(f"Loaded {len(qrels)} qrels entries.")
    print(f"Evaluating {len(query_ids)} queries.")

    results = []
    total_configs = len(RETRIEVER_OPTIONS) * len(DENSE_MODELS) * len(QUERY_EXPANSION_OPTIONS) * len(TOP_K_OPTIONS)
    total_questions = len(query_ids)
    total_runs = total_configs * total_questions
    overall_bar = tqdm(total=total_runs, desc="Overall Progress")
    start_time = time.time()
    model_cache = {}

    # Prepare all chunks (passages) as your app would
    all_chunks = []
    for doc_id, doc in corpus.items():
        chunk = {
            "id": doc_id,
            "text": doc["text"],
            "section": doc.get("title", ""),
            "document": doc_id
        }
        all_chunks.append(chunk)
    print(f"Total chunks prepared: {len(all_chunks)}")

    for retriever_type in RETRIEVER_OPTIONS:
        for dense_model in DENSE_MODELS:
            for query_expansion in QUERY_EXPANSION_OPTIONS:
                for top_k in TOP_K_OPTIONS:
                    print(f"Config: retriever={retriever_type}, dense_model={dense_model}, query_expansion={query_expansion}, top_k={top_k}")
                    for qid in query_ids:
                        question = queries[qid]
                        expected_docids = qrels.get(qid, set())
                        print(f"\nQuery ID: {qid}")
                        print(f"Question: {question}")
                        print(f"Expected docids: {expected_docids}")

                        # --- Set up retriever ---
                        index_builder = IndexBuilder()
                        index_builder.retrievers["bm25"].add_documents(all_chunks)
                        index_builder.retrievers["dense"].add_documents(all_chunks)
                        index_builder.retrievers["hybrid"].add_documents(all_chunks)
                        if retriever_type == "bm25":
                            retriever = index_builder.get_retriever("bm25")
                        elif retriever_type == "dense":
                            if dense_model not in model_cache:
                                model_cache[dense_model] = DenseRetriever(dense_model, device='cpu')
                                model_cache[dense_model].add_documents(all_chunks)
                            retriever = model_cache[dense_model]
                        elif retriever_type == "hybrid":
                            if dense_model not in model_cache:
                                model_cache[dense_model] = DenseRetriever(dense_model, device='cpu')
                                model_cache[dense_model].add_documents(all_chunks)
                            retriever = index_builder.get_retriever("hybrid")
                            retriever.dense_retriever = model_cache[dense_model]
                        else:
                            raise ValueError(f"Unknown retriever type: {retriever_type}")

                        # --- Query expansion ---
                        queries_to_run = [question]
                        if query_expansion:
                            queries_to_run += generate_paraphrases(question, num_return_sequences=3)

                        # --- Retrieve for each query, merge and deduplicate ---
                        retrieved = []
                        for q in queries_to_run:
                            if retriever_type == "hybrid":
                                chunks_ = retriever.retrieve(q, top_k=top_k)
                            elif retriever_type == "dense":
                                chunk_score_pairs = retriever.retrieve(q, top_k=top_k)
                                chunks_ = [chunk for chunk, _ in chunk_score_pairs]
                            else:
                                chunk_score_pairs = retriever.retrieve(q, top_k=top_k)
                                chunks_ = [chunk for chunk, _ in chunk_score_pairs]
                            retrieved.extend(chunks_)
                        # Deduplicate by chunk id
                        unique_chunks = []
                        seen_ids = set()
                        for chunk in retrieved:
                            cid = chunk.get("id", chunk.get("text"))
                            if cid not in seen_ids:
                                unique_chunks.append(chunk)
                                seen_ids.add(cid)
                        retrieved_chunks = unique_chunks[:top_k]

                        # --- Debug: Print retrieved chunk IDs ---
                        print(f"Retrieved chunk IDs: {[chunk['id'] for chunk in retrieved_chunks]}")

                        # --- Metrics ---
                        hit = False
                        rank = None
                        for i, chunk in enumerate(retrieved_chunks):
                            if chunk["id"] in expected_docids:
                                hit = True
                                rank = i + 1
                                print(f"Hit at rank {rank} (chunk id: {chunk['id']})")
                                break

                        cos_sim = None
                        if retriever_type in ["dense", "hybrid"] and retrieved_chunks:
                            model = model_cache[dense_model].model
                            ref_docid = next(iter(expected_docids)) if expected_docids else None
                            ref_text = corpus[ref_docid]["text"] if ref_docid else ""
                            cos_sim = cosine_sim(ref_text, retrieved_chunks[0]["text"], model)

                        recall_hit = False
                        for chunk in unique_chunks[:RECALL_K]:
                            if chunk["id"] in expected_docids:
                                recall_hit = True
                                break

                        latency = 0  # (You can time the retrieval if you want)
                        mem = psutil.Process().memory_info().rss / 1024 ** 2

                        results.append({
                            "pdf": "nfcorpus",
                            "question": question,
                            "retriever": retriever_type,
                            "dense_model": dense_model,
                            "query_expansion": query_expansion,
                            "top_k": top_k,
                            "hit": hit,
                            "rank": rank if rank is not None else "",
                            "latency": latency,
                            "cosine_sim": cos_sim if cos_sim is not None else "",
                            "recall@k": recall_hit,
                            "memory_mb": mem,
                        })
                        overall_bar.update(1)
                    pd.DataFrame(results).to_csv(args.output, index=False)

    overall_bar.close()
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved per-question results to {args.output}")

    def mrr(ranks):
        ranks = [r for r in ranks if r != ""]
        if not ranks:
            return 0.0
        return np.mean([1.0 / r for r in ranks])

    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    print("First few rows:\n", df.head())

    summary = (
        df.groupby(["retriever", "dense_model", "query_expansion", "top_k"])
        .agg(
            accuracy=("hit", "mean"),
            mrr=("rank", mrr),
            recall_at_k=("recall@k", "mean"),
            avg_rank=("rank", lambda x: np.mean([r for r in x if r != ""])),
            avg_latency=("latency", "mean"),
            avg_cosine_sim=("cosine_sim", lambda x: np.mean([v for v in x if v != ""])),
            avg_memory_mb=("memory_mb", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.summary, index=False)
    print(f"Saved per-configuration summary to {args.summary}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary, x="top_k", y="accuracy", hue="retriever")
        plt.title("Retrieval Accuracy by Retriever and top_k")
        plt.savefig("beir_eval_summary.png")
        print("Saved summary plot to beir_eval_summary.png")
    except ImportError:
        print("matplotlib/seaborn not installed, skipping plot.")

if __name__ == "__main__":
    main()
