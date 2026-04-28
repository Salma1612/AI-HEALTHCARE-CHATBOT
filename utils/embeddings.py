"""
utils/embeddings.py
-------------------
Generates vector embeddings using sentence-transformers.
Uses 'all-MiniLM-L6-v2' — a lightweight, fast, high-quality model.
"""

from langchain_huggingface import HuggingFaceEmbeddings
import torch


# Singleton — avoids reloading the model on every call
_embedding_model = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load and cache the HuggingFace embedding model.
    Uses CPU by default for compatibility with low-RAM machines.

    Returns:
        HuggingFaceEmbeddings instance (LangChain compatible)
    """
    global _embedding_model

    if _embedding_model is None:
        print(f" Loading embedding model: {model_name}")

        # Detect device — use CUDA if available, else CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Running on: {device.upper()}")

        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,  # Cosine similarity friendly
                "batch_size": 32               # Batch processing for speed
            }
        )
        print(f" Embedding model loaded successfully.\n")

    return _embedding_model


def embed_query(query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list:
    """
    Convert a single query string into an embedding vector.

    Args:
        query: User's medical question
    Returns:
        List of floats (embedding vector)
    """
    model = get_embedding_model(model_name)
    return model.embed_query(query)
