"""
PART 1 OF THE RAG SYSTEM — Free/Local Models Edition (Python 3.13 Compatible)
==============================================================================
Replaces OpenAI with:
  - Embeddings : sentence-transformers (HuggingFace, runs 100% locally)
  - Vector DB  : ChromaDB (unchanged)

Python 3.13 Fix:
  - Pins transformers to a version compatible with Python 3.13's importlib.metadata
  - Adds workaround for importlib.metadata.packages_distributions() bug

Install dependencies:
    pip install "transformers==4.40.2" langchain langchain-community langchain-chroma \
                langchain-huggingface "sentence-transformers==2.7.0" \
                chromadb python-dotenv
"""

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

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# Model choice
# ---------------------------------------------------------------------------
# "all-MiniLM-L6-v2"      – fast, lightweight, great for most RAG tasks (~80 MB)
# "all-mpnet-base-v2"     – slightly larger but higher quality            (~420 MB)
# "BAAI/bge-small-en-v1.5"– state-of-the-art small model                 (~130 MB)
# All models are downloaded automatically on first run from HuggingFace.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# 1. Load documents
# ---------------------------------------------------------------------------
def load_documents(docs_path: str = "docs") -> list:
    """Load all .txt files from the docs directory."""
    print(f"Loading documents from '{docs_path}'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist. "
            "Please create it and add your .txt company files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. "
            "Please add your company documents."
        )

    print(f"\nLoaded {len(documents)} document(s).")
    for i, doc in enumerate(documents[:2]):       # preview first 2
        print(f"\nDocument {i + 1}:")
        print(f"  Source  : {doc.metadata['source']}")
        print(f"  Length  : {len(doc.page_content)} characters")
        print(f"  Preview : {doc.page_content[:120]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


# ---------------------------------------------------------------------------
# 2. Split documents into chunks
# ---------------------------------------------------------------------------
def split_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,           # small overlap avoids cutting mid-sentence
) -> list:
    """Split documents into smaller, overlapping chunks."""
    print("\nSplitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",                 # prefer splitting on newlines
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunk(s).")

    for i, chunk in enumerate(chunks[:2]):        # preview first 2
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source : {chunk.metadata['source']}")
        print(f"Length : {len(chunk.page_content)} characters")
        print(f"Content:\n{chunk.page_content}")
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunk(s).")

    return chunks


# ---------------------------------------------------------------------------
# 3. Create vector store (free local embeddings)
# ---------------------------------------------------------------------------
def create_vector_store(
    chunks: list,
    persist_directory: str = "db/chroma_db",
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Chroma:
    """
    Embed chunks with a local HuggingFace model and store them in ChromaDB.

    No API key required — the model is downloaded once and cached locally
    (default: ~/.cache/huggingface/hub/).
    """
    print(f"\nCreating embeddings with '{model_name}' (local, free)...")
    print("(First run will download the model — this may take a moment.)\n")

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},                # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity works best normalised
    )

    print("--- Creating ChromaDB vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Vector store created successfully ---")
    print(f"Persisted to: {persist_directory}")

    return vectorstore


# ---------------------------------------------------------------------------
# 4. Quick smoke-test: verify retrieval works
# ---------------------------------------------------------------------------
def smoke_test(vectorstore: Chroma, query: str = "company overview") -> None:
    """Run a quick similarity search to confirm the store is working."""
    print(f"\n[Smoke test] Querying: '{query}'")
    results = vectorstore.similarity_search(query, k=2)
    if results:
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}: {doc.page_content[:200]}...")
    else:
        print("No results found — check that your docs contain relevant content.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== RAG Part 1: Ingestion Pipeline (Free Local Models) ===\n")

    documents   = load_documents(docs_path="docs")
    chunks      = split_documents(documents)
    vectorstore = create_vector_store(chunks)

    # Optional: confirm everything works end-to-end
    smoke_test(vectorstore)

    print("\n✅ Ingestion complete. Vector store is ready for querying.")


if __name__ == "__main__":
    main()