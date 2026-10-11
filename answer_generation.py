# ---------------------------------------------------------------------------
# Python 3.13 workaround — must be applied BEFORE importing transformers
# ---------------------------------------------------------------------------
import importlib.metadata
import importlib.metadata._functools  # noqa: F401

_original_packages_distributions = importlib.metadata.packages_distributions

def _safe_packages_distributions():
    try:
        return _original_packages_distributions()
    except Exception:
        return {}

importlib.metadata.packages_distributions = _safe_packages_distributions
# ---------------------------------------------------------------------------

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # suppress symlinks warning on Windows

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

persistent_directory = "db/chroma_db"

# Load embeddings and vector store — must match ingestion model!
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""You are a helpful assistant. Answer the question based only on the documents below.

Question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Answer:"""

# ---------------------------------------------------------------------------
# Free local LLM — uses text-generation (compatible with latest transformers)
# Model: TinyLlama (~600 MB, downloads automatically)
# Other options:
#   "facebook/opt-125m"   — very small, ~250 MB, lower quality
#   "facebook/opt-350m"   — better quality, ~700 MB
# ---------------------------------------------------------------------------
print("\nLoading local LLM (first run will download the model)...")
local_pipeline = pipeline(
    "text-generation",
    model="facebook/opt-125m",   # only ~250 MB
    max_new_tokens=256,
    do_sample=False,
    temperature=None,
    top_p=None,
)

# Invoke the model
result = local_pipeline(combined_input)

# Extract only the generated answer (strip the prompt from output)
full_output = result[0]["generated_text"]
answer = full_output[len(combined_input):].strip()

print("\n--- Generated Response ---")
print("Content only:")
print(answer)