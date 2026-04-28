"""
utils/text_splitter.py
----------------------
Splits long medical texts into overlapping chunks for embedding.
Uses LangChain's RecursiveCharacterTextSplitter.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict


def split_documents(
    docs: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
    """
    Split loaded PDF documents into overlapping text chunks.

    Args:
        docs        : List of dicts from pdf_loader (with 'text' and 'source' keys)
        chunk_size  : Number of characters per chunk (500, 800, or 1000 recommended)
        chunk_overlap: Characters shared between consecutive chunks

    Returns:
        List of LangChain Document objects with page_content + metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # These separators help preserve paragraph structure
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    all_chunks: List[Document] = []

    for doc in docs:
        raw_text = doc.get("text", "").strip()
        source   = doc.get("source", "unknown.pdf")

        if not raw_text:
            continue

        # Split into chunks
        chunks = splitter.split_text(raw_text)

        for i, chunk in enumerate(chunks):
            langchain_doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "chunk_total": len(chunks)
                }
            )
            all_chunks.append(langchain_doc)

    print(f" Text splitting complete: {len(all_chunks)} chunks created "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")

    return all_chunks


def get_chunk_stats(chunks: List[Document]) -> Dict:
    """
    Return basic stats about the chunk distribution.
    Useful for evaluation and reporting.
    """
    lengths = [len(c.page_content) for c in chunks]
    sources = list(set(c.metadata.get("source", "?") for c in chunks))

    return {
        "total_chunks": len(chunks),
        "avg_chunk_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chunk_len": min(lengths) if lengths else 0,
        "max_chunk_len": max(lengths) if lengths else 0,
        "sources": sources,
        "source_count": len(sources)
    }
