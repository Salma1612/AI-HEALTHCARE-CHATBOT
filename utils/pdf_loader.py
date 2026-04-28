"""
utils/pdf_loader.py
-------------------
Handles loading and extracting text from PDF files.
Supports both PyPDF and PDFPlumber as fallback.
"""

import os
import re
import pdfplumber
from pypdf import PdfReader
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing unwanted symbols,
    extra whitespace, and non-printable characters.
    """
    if not text:
        return ""
    # Remove non-ASCII characters except common punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove multiple spaces / newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers pattern like "Page 1 of 10"
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    return text.strip()


def load_pdf_pypdf(file_path: str) -> str:
    """
    Load text from a PDF using PyPDF.
    Returns concatenated text from all pages.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"[PyPDF] Error reading {file_path}: {e}")
    return text


def load_pdf_pdfplumber(file_path: str) -> str:
    """
    Load text from a PDF using PDFPlumber (better for tables).
    Returns concatenated text from all pages.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[PDFPlumber] Error reading {file_path}: {e}")
    return text


def load_single_pdf(file_path: str) -> Dict:
    """
    Load a single PDF file and return its metadata + text.
    Tries PyPDF first, falls back to PDFPlumber.
    """
    filename = os.path.basename(file_path)
    print(f"  Loading: {filename}")

    # Try PyPDF first
    raw_text = load_pdf_pypdf(file_path)

    # If PyPDF gives poor results, try PDFPlumber
    if len(raw_text.strip()) < 100:
        print(f"  PyPDF gave sparse text, trying PDFPlumber...")
        raw_text = load_pdf_pdfplumber(file_path)

    clean = clean_text(raw_text)

    return {
        "source": filename,
        "file_path": file_path,
        "text": clean,
        "char_count": len(clean)
    }


def load_all_pdfs(data_folder: str) -> List[Dict]:
    """
    Load all PDF files from a folder.
    Returns list of dicts with text + metadata.
    """
    results = []

    if not os.path.exists(data_folder):
        print(f"[ERROR] Data folder not found: {data_folder}")
        return results

    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {data_folder}")
        return results

    print(f"\n Found {len(pdf_files)} PDF file(s) in '{data_folder}':")
    for pdf_file in sorted(pdf_files):
        full_path = os.path.join(data_folder, pdf_file)
        doc = load_single_pdf(full_path)
        if doc["char_count"] > 50:
            results.append(doc)
            print(f"  ✔ {doc['source']} — {doc['char_count']} characters extracted")
        else:
            print(f"  ✗ {doc['source']} — Too little text, skipping")

    print(f"\n Successfully loaded {len(results)} document(s).\n")
    return results
