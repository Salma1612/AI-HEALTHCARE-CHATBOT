"""
train_index.py
--------------
One-time script to:
  1. Load all PDFs from /data folder
  2. Split text into chunks
  3. Generate embeddings
  4. Save FAISS index to /faiss_index

Run this BEFORE starting the chatbot app.
Usage:
    python train_index.py
    python train_index.py --chunk_size 1000 --overlap 100
"""

import os
import sys
import argparse
import json
import time

# ── Local imports ──────────────────────────────────────────────────────────────
from utils.pdf_loader    import load_all_pdfs
from utils.text_splitter import split_documents, get_chunk_stats
from utils.embeddings    import get_embedding_model
from utils.vector_store  import build_faiss_index, save_faiss_index


# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FOLDER       = "data"
FAISS_INDEX_PATH  = "faiss_index"
STATS_OUTPUT_FILE = "faiss_index/index_stats.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from medical PDFs")
    parser.add_argument("--chunk_size",  type=int, default=800,
                        help="Characters per chunk (default: 800)")
    parser.add_argument("--overlap",     type=int, default=80,
                        help="Overlap characters between chunks (default: 80)")
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER,
                        help="Path to folder with PDFs (default: data/)")
    parser.add_argument("--index_path",  type=str, default=FAISS_INDEX_PATH,
                        help="Where to save FAISS index (default: faiss_index/)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  AI Medical Chatbot — FAISS Index Builder")
    print("=" * 60)
    print(f"  Data folder  : {args.data_folder}")
    print(f"  Chunk size   : {args.chunk_size}")
    print(f"  Overlap      : {args.overlap}")
    print(f"  Index output : {args.index_path}")
    print("=" * 60 + "\n")

    start_total = time.time()

    # ── Step 1: Load PDFs ───────────────────────────────────────────────────
    print("STEP 1: Loading PDF documents...")
    docs = load_all_pdfs(args.data_folder)

    if not docs:
        print("\n[ERROR] No documents loaded. Please add PDF files to the data/ folder.")
        print("  Recommended datasets: MSD Manual, WHO disease fact sheets,")
        print("  PubMed Open Access, or MedlinePlus PDFs.\n")
        sys.exit(1)

    # ── Step 2: Split into chunks ───────────────────────────────────────────
    print("STEP 2: Splitting documents into chunks...")
    chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.overlap)

    if not chunks:
        print("[ERROR] No chunks created. Check your PDF content.")
        sys.exit(1)

    stats = get_chunk_stats(chunks)
    print(f"\n  Chunk Statistics:")
    print(f"    Total chunks     : {stats['total_chunks']}")
    print(f"    Avg chunk length : {stats['avg_chunk_len']} chars")
    print(f"    Min / Max        : {stats['min_chunk_len']} / {stats['max_chunk_len']} chars")
    print(f"    Source files     : {stats['source_count']}")
    for s in stats['sources']:
        print(f"      - {s}")
    print()

    # ── Step 3: Load embedding model ────────────────────────────────────────
    print("STEP 3: Loading embedding model...")
    embedding_model = get_embedding_model()

    # ── Step 4: Build FAISS index ───────────────────────────────────────────
    print("STEP 4: Building FAISS vector index...")
    vector_store = build_faiss_index(chunks, embedding_model)

    # ── Step 5: Save index ──────────────────────────────────────────────────
    print("STEP 5: Saving FAISS index...")
    save_faiss_index(vector_store, path=args.index_path)

    # ── Save stats for evaluation ───────────────────────────────────────────
    stats["chunk_size"]  = args.chunk_size
    stats["overlap"]     = args.overlap
    stats["build_time_sec"] = round(time.time() - start_total, 2)

    os.makedirs(args.index_path, exist_ok=True)
    with open(STATS_OUTPUT_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Done ────────────────────────────────────────────────────────────────
    total_time = round(time.time() - start_total, 2)
    print("\n" + "=" * 60)
    print(f"  Index built successfully in {total_time}s!")
    print(f"  Run 'streamlit run app.py' to start the chatbot.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
