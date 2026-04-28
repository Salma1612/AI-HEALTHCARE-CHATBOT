"""
utils/vector_store.py
---------------------
Handles building, saving, and loading the FAISS vector index.
FAISS provides fast approximate nearest-neighbor search.
"""

import os
import time
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Tuple, Optional


FAISS_INDEX_PATH = "faiss_index"   # Folder where index is saved


def build_faiss_index(chunks: List[Document], embedding_model) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    Args:
        chunks         : List of LangChain Document objects
        embedding_model: HuggingFaceEmbeddings instance

    Returns:
        FAISS vector store object
    """
    print(f" Building FAISS index from {len(chunks)} chunks...")
    start = time.time()

    vector_store = FAISS.from_documents(chunks, embedding_model)

    elapsed = round(time.time() - start, 2)
    print(f" FAISS index built in {elapsed}s\n")

    return vector_store


def save_faiss_index(vector_store: FAISS, path: str = FAISS_INDEX_PATH):
    """
    Save the FAISS index to disk for reuse.
    Saves two files: index.faiss and index.pkl
    """
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    print(f" FAISS index saved to: {path}/")


def load_faiss_index(embedding_model, path: str = FAISS_INDEX_PATH) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk.

    Returns:
        FAISS object if found, else None
    """
    faiss_file = os.path.join(path, "index.faiss")

    if not os.path.exists(faiss_file):
        print(f"[INFO] No saved FAISS index found at: {path}")
        return None

    print(f" Loading FAISS index from: {path}/")
    vector_store = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True  # Required by LangChain >= 0.1
    )
    print(f" FAISS index loaded successfully.\n")
    return vector_store


def retrieve_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4
) -> List[Tuple[Document, float]]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Args:
        vector_store: FAISS index
        query       : User question
        top_k       : Number of chunks to retrieve (default: 4)

    Returns:
        List of (Document, similarity_score) tuples, sorted by score descending
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)

    # Lower L2 score = more similar (convert to 0-1 confidence)
    # For normalized embeddings, scores are typically 0.0–2.0 range
    formatted = []
    for doc, score in results:
        # Convert L2 distance to a 0–1 similarity score
        confidence = round(max(0.0, 1.0 - score / 2.0), 3)
        formatted.append((doc, confidence))

    # Sort by confidence descending
    formatted.sort(key=lambda x: x[1], reverse=True)
    return formatted
