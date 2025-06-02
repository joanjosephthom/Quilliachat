import streamlit as st
import os
import pickle
import json

CHUNKED_DIR = "./chunked_data"  # Directory with .chunks.pkl files
OUTPUT_FILE = "labeled_eval_dataset.jsonl"

st.title("Quillia Advanced Chunk Labeling Tool")

# --- Load PDFs and Chunks ---
pdf_files = [f for f in os.listdir(CHUNKED_DIR) if f.endswith(".chunks.pkl")]
pdf_choice = st.selectbox("Select PDF", pdf_files)

with open(os.path.join(CHUNKED_DIR, pdf_choice), "rb") as f:
    chunks = pickle.load(f)

# --- Load existing labels ---
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f]
else:
    labels = []

# --- Labeling UI ---
question = st.text_area("Enter the question you want to label:")
answer_snippet = st.text_input("Enter the answer snippet (optional, for highlighting):")

# --- Search/filter chunks ---
search_term = st.text_input("Search/filter chunks (by keyword):")
if search_term:
    filtered_chunks = [chunk for chunk in chunks if search_term.lower() in chunk["text"].lower()]
else:
    filtered_chunks = chunks

# --- Multi-chunk selection ---
chunk_options = [
    f"Chunk {i} (Section: {chunk.get('section','')}, Page: {chunk.get('page','')})"
    for i, chunk in enumerate(filtered_chunks)
]
selected_indices = st.multiselect(
    "Select one or more chunks that contain the answer:",
    options=list(range(len(filtered_chunks))),
    format_func=lambda i: chunk_options[i]
)

# --- Highlight answer snippet in chunk text ---
def highlight_text(text, snippet):
    if not snippet or snippet not in text:
        return text
    return text.replace(snippet, f"<mark style='background-color: #ffe066'>{snippet}</mark>")

if question and selected_indices:
    st.write("**Selected Chunks Preview:**")
    for i in selected_indices:
        chunk = filtered_chunks[i]
        preview = chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else "")
        highlighted = highlight_text(preview, answer_snippet)
        st.markdown(f"<div style='border:1px solid #eee; border-radius:8px; padding:8px; margin-bottom:8px;'>{highlighted}</div>", unsafe_allow_html=True)
        with st.expander("Show full chunk"):
            full_highlighted = highlight_text(chunk["text"], answer_snippet)
            st.markdown(f"<div style='border:1px solid #eee; border-radius:8px; padding:8px; margin-bottom:8px;'>{full_highlighted}</div>", unsafe_allow_html=True)

if st.button("Save Label", disabled=not (question and selected_indices)):
    label = {
        "pdf": pdf_choice.replace(".chunks.pkl", ".pdf"),
        "question": question,
        "expected_chunk_ids": [filtered_chunks[i].get("id", "") for i in selected_indices],
        "expected_chunks": [filtered_chunks[i]["text"] for i in selected_indices],
        "answer_snippet": answer_snippet,
    }
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        out.write(json.dumps(label, ensure_ascii=False) + "\n")
    st.success("Label saved!")

# --- Edit/Delete Labels ---
st.markdown("---")
st.header("Existing Labels for This PDF")
pdf_labels = [l for l in labels if l["pdf"] == pdf_choice.replace(".chunks.pkl", ".pdf")]
for idx, label in enumerate(pdf_labels):
    st.markdown(f"**Q{idx+1}:** {label['question']}")
    for cidx, chunk_text in enumerate(label["expected_chunks"]):
        st.markdown(f"- Chunk {cidx}: {chunk_text[:120]}{'...' if len(chunk_text)>120 else ''}")
    if st.button(f"Delete Label {idx+1}", key=f"delete_{idx}"):
        labels.remove(label)
        # Rewrite the file without this label
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            for l in labels:
                out.write(json.dumps(l, ensure_ascii=False) + "\n")
        st.success("Label deleted. Please refresh the page.")

st.markdown("---")
st.info("Tip: Use the search box to quickly find relevant chunks. You can select multiple chunks if the answer spans more than one. You can also edit or delete labels below.")
