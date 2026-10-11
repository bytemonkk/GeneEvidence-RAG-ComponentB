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

# Redirect HuggingFace model cache to D: drive (avoids C: drive space issues)
os.environ["HF_HUB_CACHE"] = "D:/hf_cache"
os.environ["HF_HOME"]      = "D:/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # suppress Windows symlinks warning

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from transformers import pipeline

# ---------------------------------------------------------------------------
# Connect to your document database
# ---------------------------------------------------------------------------
persistent_directory = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",            # must match ingestion model
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

# ---------------------------------------------------------------------------
# Load local LLM (downloaded to D:/hf_cache on first run, ~600 MB)
# ---------------------------------------------------------------------------
print("Loading local LLM (first run will download to D:/hf_cache)...")
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256,
    do_sample=False,
    temperature=None,
    top_p=None,
)
print("LLM ready.\n")

# ---------------------------------------------------------------------------
# Store conversation as messages
# ---------------------------------------------------------------------------
chat_history = []


def _llm_invoke(prompt: str) -> str:
    """Helper: call the local pipeline and return the generated text only."""
    result = llm(prompt)
    full   = result[0]["generated_text"]
    return full[len(prompt):].strip()


def ask_question(user_question: str) -> str:
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question standalone using conversation history
    if chat_history:
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in chat_history
        )
        rewrite_prompt = (
            f"Given this chat history:\n{history_text}\n\n"
            f"Rewrite the following question to be standalone and searchable. "
            f"Return only the rewritten question.\n\nQuestion: {user_question}\nRewritten:"
        )
        search_question = _llm_invoke(rewrite_prompt).split("\n")[0].strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs      = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant document(s):")
    for i, doc in enumerate(docs, 1):
        lines   = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Build final prompt with history + documents
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in chat_history
    ) if chat_history else ""

    docs_text = "\n".join([f"- {doc.page_content}" for doc in docs])

    combined_prompt = (
        f"You are a helpful assistant that answers questions based on provided documents.\n\n"
        + (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
        + f"Documents:\n{docs_text}\n\n"
        f"Based only on the documents above, answer this question: {user_question}\n"
        f"If the answer is not in the documents, say 'I don't have enough information to answer that question based on the provided documents.'\n\n"
        f"Answer:"
    )

    # Step 4: Get the answer
    answer = _llm_invoke(combined_prompt)

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


# ---------------------------------------------------------------------------
# Simple chat loop
# ---------------------------------------------------------------------------
def start_chat() -> None:
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()