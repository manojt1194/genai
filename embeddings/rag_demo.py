import os
from typing import List

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Configuration
# -------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


# -------------------------------------------------
# Sample knowledge base
# -------------------------------------------------
documents = [
    """
    FastAPI is a modern Python web framework.

    It automatically generates API documentation.

    It supports asynchronous programming.

    It is commonly used in GenAI applications.
    """,
    """
    LangChain is a framework used to build LLM-powered applications.

    It supports RAG, agents, memory, and tool calling.
    """,
    """
    FAISS is a vector similarity search library.

    It allows efficient retrieval of embeddings.
    """,
]


# -------------------------------------------------
# Step 1: Chunking
# -------------------------------------------------
def chunk_documents(docs: List[str]) -> List[str]:
    chunks = []
    for doc in docs:
        # Split by paragraph and clean up spaces
        parts = [part.strip() for part in doc.strip().split("\n\n") if part.strip()]
        chunks.extend(parts)
    return chunks


# -------------------------------------------------
# Step 2: Build embeddings
# -------------------------------------------------
def build_embeddings(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(chunks)
    return np.array(embeddings, dtype="float32")


# -------------------------------------------------
# Step 3: Build FAISS index
# -------------------------------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# -------------------------------------------------
# Step 4: Retrieve top-k relevant chunks
# -------------------------------------------------
def retrieve_context(
    query: str,
    chunks: List[str],
    model: SentenceTransformer,
    index: faiss.Index,
    k: int = 3,
) -> List[str]:
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results


# -------------------------------------------------
# Step 5: Call local LLM
# -------------------------------------------------
def call_local_llm(messages: list) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running? Start it with: ollama serve"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Ollama API error: {e.response.status_code} {e.response.reason}\n"
            f"URL: {url}\n"
            f"Model: {OLLAMA_MODEL}\n"
            f"Response: {e.response.text}"
        )
    
    data = response.json()
    return data["message"]["content"]


# -------------------------------------------------
# Step 6: Build RAG answer
# -------------------------------------------------
def ask_rag(
    query: str,
    chunks: List[str],
    model: SentenceTransformer,
    index: faiss.Index,
) -> str:
    context_chunks = retrieve_context(
        query=query,
        chunks=chunks,
        model=model,
        index=index,
        k=3,
    )

    context_text = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question using only the context below.

Context:
{context_text}

Question:
{query}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use only the provided context.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    print('messages', messages)

    return call_local_llm(messages)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    # print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # print("Chunking documents...")
    chunks = chunk_documents(documents)

    # print("Chunks:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")

    # print("\nBuilding embeddings...")
    chunk_embeddings = build_embeddings(chunks, model)

    # print("Building FAISS index...")
    index = build_faiss_index(chunk_embeddings)

    print(f"Vectors stored in FAISS: {index.ntotal}")

    # Sample question
    query = "How does FastAPI create documentation?"

    print(f"\nQuery: {query}")
    answer = ask_rag(query, chunks, model, index)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()