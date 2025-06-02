import os
import json
import time
import argparse
import pickle
import logging
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from indexers.index_builder import IndexBuilder
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
from utils import generate_paraphrases, build_token_limited_context
from config import EMBEDDING_MODELS

RETRIEVER_OPTIONS = ["bm25", "dense", "hybrid"]
DENSE_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
QUERY_EXPANSION_OPTIONS = [False, True]
TOP_K_OPTIONS = [3, 5, 10]
RECALL_K = 10

logging.basicConfig(
    filename='eval.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def log_mem_usage():
    mem = psutil.Process().memory_info().rss / 1024 ** 2
    logging.info(f"Memory usage (MB): {mem:.2f}")

def load_eval_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_sim(text1, text2, model):
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])

def cache_chunks(pdf_path, pdf_name, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"chunks_{pdf_name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    index_builder = IndexBuilder()
    with open(pdf_path, "rb") as f:
        chunks = index_builder.chunker.extract_chunks_from_pdf(pdf_path, pdf_name)
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    return chunks

def run_retrieval(index_builder, retriever_type, dense_model, query_expansion, top_k, question, expected_chunks, chunks, model_cache):
    if retriever_type == "bm25":
        retriever = index_builder.get_retriever("bm25")
    elif retriever_type == "dense":
        if dense_model not in model_cache:
            model_cache[dense_model] = DenseRetriever(dense_model, device='cpu')
        retriever = model_cache[dense_model]
        retriever.add_documents(chunks)  # <-- ADD THIS LINE
    elif retriever_type == "hybrid":
        if dense_model not in model_cache:
            model_cache[dense_model] = DenseRetriever(dense_model, device='cpu')
        retriever = index_builder.get_retriever("hybrid")
        retriever.dense_retriever = model_cache[dense_model]
        retriever.dense_retriever.add_documents(chunks)  # <-- ADD THIS LINE
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    queries = [question]
    if query_expansion:
        queries += generate_paraphrases(question, num_return_sequences=3)

    all_chunks = []
    for q in queries:
        if retriever_type == "hybrid":
            chunks_ = retriever.retrieve(q, top_k=top_k)
        elif retriever_type == "dense":
            chunk_score_pairs = retriever.retrieve(q, top_k=top_k)
            chunks_ = [chunk for chunk, _ in chunk_score_pairs]
        else:
            chunk_score_pairs = retriever.retrieve(q, top_k=top_k)
            chunks_ = [chunk for chunk, _ in chunk_score_pairs]
        all_chunks.extend(chunks_)
    unique_chunks = []
    seen_ids = set()
    for chunk in all_chunks:
        cid = chunk.get("id", chunk.get("text"))
        if cid not in seen_ids:
            unique_chunks.append(chunk)
            seen_ids.add(cid)
    retrieved_chunks = unique_chunks[:top_k]

    # --- Multi-chunk hit/rank check ---
    hit = False
    rank = None
    for i, chunk in enumerate(retrieved_chunks):
        for expected_chunk in expected_chunks:
            if expected_chunk.strip() in chunk["text"]:
                hit = True
                rank = i + 1
                break
        if hit:
            break

    cos_sim = None
    if retriever_type in ["dense", "hybrid"] and retrieved_chunks:
        model = model_cache[dense_model].model
        cos_sim = cosine_sim(expected_chunks[0], retrieved_chunks[0]["text"], model)

    recall_hit = False
    for chunk in unique_chunks[:RECALL_K]:
        for expected_chunk in expected_chunks:
            if expected_chunk.strip() in chunk["text"]:
                recall_hit = True
                break
        if recall_hit:
            break

    return hit, rank, retrieved_chunks, cos_sim, recall_hit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation JSON")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDFs")
    parser.add_argument("--output", type=str, default="retrieval_eval_results.csv", help="Output CSV for per-question results")
    parser.add_argument("--summary", type=str, default="retrieval_eval_summary.csv", help="Output CSV for per-config summary")
    parser.add_argument("--pdf_subset", nargs="*", help="List of PDF filenames to include (for quick tests)")
    parser.add_argument("--question_subset", nargs="*", help="List of question strings to include (for quick tests)")
    parser.add_argument("--max_questions", type=int, help="Max number of questions to run (for quick tests)")
    args = parser.parse_args()

    eval_data = load_eval_dataset(args.dataset)
    if args.pdf_subset:
        eval_data = [q for q in eval_data if q["pdf"] in args.pdf_subset]
    if args.question_subset:
        eval_data = [q for q in eval_data if q["question"] in args.question_subset]
    if args.max_questions:
        eval_data = eval_data[:args.max_questions]

    total_configs = len(RETRIEVER_OPTIONS) * len(DENSE_MODELS) * len(QUERY_EXPANSION_OPTIONS) * len(TOP_K_OPTIONS)
    total_questions = len(eval_data)
    total_runs = total_configs * total_questions

    overall_bar = tqdm(total=total_runs, desc="Overall Progress")
    start_time = time.time()
    results = []
    model_cache = {}
    chunk_cache = {}

    for retriever_type in RETRIEVER_OPTIONS:
        for dense_model in DENSE_MODELS:
            for query_expansion in QUERY_EXPANSION_OPTIONS:
                for top_k in TOP_K_OPTIONS:
                    logging.info(f"Testing config: retriever={retriever_type}, dense_model={dense_model}, query_expansion={query_expansion}, top_k={top_k}")
                    for entry in eval_data:
                        pdf = entry["pdf"]
                        question = entry["question"]
                        expected_chunks = entry.get("expected_chunks", [])
                        pdf_path = os.path.join(args.pdf_dir, pdf)
                        if not os.path.exists(pdf_path):
                            logging.warning(f"PDF not found: {pdf_path}")
                            overall_bar.update(1)
                            continue
                        chunk_key = (pdf, )
                        if chunk_key not in chunk_cache:
                            index_builder = IndexBuilder()
                            chunks = index_builder.process_file_bytes(open(pdf_path, "rb").read(), pdf)
                            chunk_cache[chunk_key] = (index_builder, chunks)
                        else:
                            index_builder, chunks = chunk_cache[chunk_key]
                        start = time.time()
                        hit, rank, retrieved_chunks, cos_sim, recall_hit = run_retrieval(
                            index_builder, retriever_type, dense_model, query_expansion, top_k, question, expected_chunks, chunks, model_cache
                        )
                        latency = time.time() - start
                        mem = psutil.Process().memory_info().rss / 1024 ** 2
                        results.append({
                            "pdf": pdf,
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
                    logging.info(f"Saved intermediate results to {args.output}")

    overall_bar.close()
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes")
    logging.info(f"Total elapsed time: {elapsed/60:.2f} minutes")

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

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="pastel")

    # 1. Accuracy by retriever, top_k, and query_expansion (facet by dense_model)
    g = sns.catplot(
        data=summary,
        x="top_k",
        y="accuracy",
        hue="query_expansion",
        col="dense_model",
        kind="bar",
        height=4,
        aspect=1.2,
        ci=None
    )
    g.fig.suptitle("Accuracy by Retriever, top_k, Query Expansion, and Dense Model", y=1.08)
    g.set_axis_labels("top_k", "Accuracy")
    g.add_legend()
    plt.savefig("accuracy_facet_dense_model.png")
    print("Saved plot: accuracy_facet_dense_model.png")

    # 2. MRR by retriever, top_k, and query_expansion (facet by dense_model)
    g = sns.catplot(
        data=summary,
        x="top_k",
        y="mrr",
        hue="query_expansion",
        col="dense_model",
        kind="bar",
        height=4,
        aspect=1.2,
        ci=None
    )
    g.fig.suptitle("MRR by Retriever, top_k, Query Expansion, and Dense Model", y=1.08)
    g.set_axis_labels("top_k", "MRR")
    g.add_legend()
    plt.savefig("mrr_facet_dense_model.png")
    print("Saved plot: mrr_facet_dense_model.png")

    # 3. Avg Rank by retriever, top_k, and query_expansion (facet by dense_model)
    g = sns.catplot(
        data=summary,
        x="top_k",
        y="avg_rank",
        hue="query_expansion",
        col="dense_model",
        kind="bar",
        height=4,
        aspect=1.2,
        ci=None
    )
    g.fig.suptitle("Avg Rank by Retriever, top_k, Query Expansion, and Dense Model", y=1.08)
    g.set_axis_labels("top_k", "Avg Rank")
    g.add_legend()
    plt.savefig("avg_rank_facet_dense_model.png")
    print("Saved plot: avg_rank_facet_dense_model.png")

    # 4. Recall@k by retriever, top_k, and query_expansion (facet by dense_model)
    g = sns.catplot(
        data=summary,
        x="top_k",
        y="recall_at_k",
        hue="query_expansion",
        col="dense_model",
        kind="bar",
        height=4,
        aspect=1.2,
        ci=None
    )
    g.fig.suptitle("Recall@k by Retriever, top_k, Query Expansion, and Dense Model", y=1.08)
    g.set_axis_labels("top_k", "Recall@k")
    g.add_legend()
    plt.savefig("recall_at_k_facet_dense_model.png")
    print("Saved plot: recall_at_k_facet_dense_model.png")

    # 5. Avg Memory Usage by retriever, top_k, and query_expansion (facet by dense_model)
    g = sns.catplot(
        data=summary,
        x="top_k",
        y="avg_memory_mb",
        hue="query_expansion",
        col="dense_model",
        kind="bar",
        height=4,
        aspect=1.2,
        ci=None
    )
    g.fig.suptitle("Avg Memory Usage by Retriever, top_k, Query Expansion, and Dense Model", y=1.08)
    g.set_axis_labels("top_k", "Avg Memory (MB)")
    g.add_legend()
    plt.savefig("avg_memory_facet_dense_model.png")
    print("Saved plot: avg_memory_facet_dense_model.png")

    # 6. Accuracy by retriever, top_k, and dense_model (facet by query_expansion)
    g = sns.catplot(
        data=summary,
        x="top_k",
        y="accuracy",
        hue="dense_model",
        col="query_expansion",
        kind="bar",
        height=4,
        aspect=1.2,
        ci=None
    )
    g.fig.suptitle("Accuracy by Retriever, top_k, Dense Model, and Query Expansion", y=1.08)
    g.set_axis_labels("top_k", "Accuracy")
    g.add_legend()
    plt.savefig("accuracy_facet_query_expansion.png")
    print("Saved plot: accuracy_facet_query_expansion.png")

if __name__ == "__main__":
    main()
