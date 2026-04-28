"""
chatbot.py
----------
Core RAG pipeline:
  1. Load FAISS index
  2. Retrieve relevant chunks for user query
  3. Build a prompt with context
  4. Generate answer using Flan-T5 (local, no OpenAI needed)
  5. Return answer + sources + confidence scores + response time
"""

import time
import torch
from typing import List, Dict, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from utils.embeddings   import get_embedding_model
from utils.vector_store import load_faiss_index, retrieve_chunks


# ── Configuration ──────────────────────────────────────────────────────────────
FAISS_INDEX_PATH  = "faiss_index"
GENERATOR_MODEL   = "google/flan-t5-base"   # Change to flan-t5-small if RAM < 4GB
TOP_K_CHUNKS      = 4                        # Number of context chunks to retrieve
MAX_NEW_TOKENS    = 256                      # Max tokens in generated answer
MIN_CONFIDENCE    = 0.20                     # Below this → suggest consulting a doctor

MEDICAL_SAFE_PROMPT = """You are a helpful medical information assistant.
Answer ONLY using the provided context. Be concise and accurate.
If the answer is not found in the context, respond with:
"I don't have specific information on this topic. Please consult a qualified doctor."
Do NOT make up information. Do NOT provide diagnosis or prescriptions.

Context:
{context}

Question: {question}

Answer:"""


# ── Singleton caches ───────────────────────────────────────────────────────────
_vector_store  = None
_generator     = None
_embed_model   = None


def load_resources(index_path: str = FAISS_INDEX_PATH):
    """
    Load embedding model + FAISS index + generator model.
    Called once at startup; subsequent calls use cached objects.
    """
    global _vector_store, _generator, _embed_model

    # Load embedding model
    if _embed_model is None:
        _embed_model = get_embedding_model()

    # Load FAISS index
    if _vector_store is None:
        _vector_store = load_faiss_index(_embed_model, path=index_path)
        if _vector_store is None:
            raise FileNotFoundError(
                "FAISS index not found. Run 'python train_index.py' first."
            )

    # Load Flan-T5 generator
    if _generator is None:
        _load_generator()

    return _vector_store, _generator


def _load_generator():
    """Load Flan-T5 text generation pipeline."""
    global _generator

    device = 0 if torch.cuda.is_available() else -1   # -1 = CPU
    device_name = "CUDA" if device == 0 else "CPU"

    print(f" Loading generator model: {GENERATOR_MODEL} on {device_name}")

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATOR_MODEL,
        torch_dtype=torch.float32,   # float32 for CPU compatibility
        low_cpu_mem_usage=True
    )

    _generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,             # Deterministic for medical safety
        temperature=1.0
    )
    print(f" Generator model loaded.\n")


def build_context(retrieved: List[Tuple]) -> str:
    """
    Concatenate retrieved chunk texts into a single context string.
    """
    context_parts = []
    for i, (doc, score) in enumerate(retrieved):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[Source {i+1}: {source}]\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def generate_answer(question: str, context: str) -> str:
    """
    Run the Flan-T5 generator with the RAG prompt.
    Returns the generated answer string.
    """
    global _generator

    prompt = MEDICAL_SAFE_PROMPT.format(
        context=context[:3500],   # Truncate to avoid token overflow
        question=question
    )

    result = _generator(prompt)
    answer = result[0]["generated_text"].strip()

    # Safety fallback for empty answers
    if not answer or len(answer) < 5:
        answer = ("I don't have enough information to answer this question accurately. "
                  "Please consult a healthcare professional.")

    return answer


def get_response(
    question: str,
    chat_history: Optional[List[Dict]] = None,
    top_k: int = TOP_K_CHUNKS
) -> Dict:
    """
    Main chatbot function — full RAG pipeline.

    Args:
        question    : User's medical question
        chat_history: List of previous {"role": ..., "content": ...} dicts
        top_k       : Number of chunks to retrieve

    Returns:
        Dict with keys:
            answer       : Generated answer string
            sources      : List of source filenames
            chunks       : List of retrieved chunk texts
            confidences  : List of similarity scores (0–1)
            response_time: Time in seconds
            low_confidence: Bool — True if best score < MIN_CONFIDENCE
    """
    start = time.time()

    # ── Load resources (cached after first call) ───────────────────────────
    vector_store, generator = load_resources()

    # ── Retrieve relevant chunks ───────────────────────────────────────────
    retrieved = retrieve_chunks(vector_store, question, top_k=top_k)

    if not retrieved:
        return {
            "answer": "I couldn't find relevant information. Please consult a doctor.",
            "sources": [],
            "chunks": [],
            "confidences": [],
            "response_time": round(time.time() - start, 2),
            "low_confidence": True
        }

    # ── Build context ──────────────────────────────────────────────────────
    context     = build_context(retrieved)
    sources     = [doc.metadata.get("source", "?") for doc, _ in retrieved]
    chunks      = [doc.page_content for doc, _ in retrieved]
    confidences = [score for _, score in retrieved]
    top_conf    = confidences[0] if confidences else 0.0

    # ── Generate answer ────────────────────────────────────────────────────
    answer = generate_answer(question, context)

    # ── Low confidence fallback ────────────────────────────────────────────
    low_confidence = top_conf < MIN_CONFIDENCE
    if low_confidence:
        answer += ("\n\n⚠️ Note: The retrieved context has low confidence for this query. "
                   "Please verify with a medical professional.")

    response_time = round(time.time() - start, 2)

    return {
        "answer"        : answer,
        "sources"       : list(dict.fromkeys(sources)),   # deduplicated
        "chunks"        : chunks,
        "confidences"   : confidences,
        "response_time" : response_time,
        "low_confidence": low_confidence
    }


def is_index_ready(index_path: str = FAISS_INDEX_PATH) -> bool:
    """Check if the FAISS index exists and is ready to use."""
    import os
    return os.path.exists(os.path.join(index_path, "index.faiss"))
