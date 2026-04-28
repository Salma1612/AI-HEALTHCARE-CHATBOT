# 🏥 AI Medical Chatbot using RAG (Retrieval-Augmented Generation)

> **Team:** SK Salma (23BCE20344) · Syed Muskan (23BCE7305)
> **Guide:** Prof. E. Sreenivasa Reddy, VIT-AP University

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Recommendation](#dataset-recommendation)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Evaluation](#evaluation)
- [Example Queries](#example-queries)
- [Disclaimer](#disclaimer)

---

## 🧠 Overview

This chatbot answers health-related questions by retrieving information from trusted medical PDF documents and generating concise answers using a local HuggingFace language model — **no OpenAI API or internet required**.

Key capabilities:
- ✅ Upload medical PDFs as knowledge base
- ✅ Fast retrieval using FAISS vector search
- ✅ Answer generation via Google Flan-T5 (runs on CPU)
- ✅ Confidence scoring per retrieved chunk
- ✅ Chat history with session memory
- ✅ Streamlit professional UI with dark theme
- ✅ Evaluation graphs and accuracy metrics

---

## 📚 Recommended Dataset for Whole Healthcare Coverage

Instead of individual disease PDFs, use **one comprehensive dataset** that covers all major health topics:

### ✅ Best Option: MSD Manual (Professional & Consumer)
- **URL**: https://www.msdmanuals.com/professional (download chapter PDFs)
- Covers: 24+ medical specialties, 10,000+ topics
- License: Free for educational use
- Format: PDF, accessible by chapter

### ✅ Option 2: WHO Disease Fact Sheets (Official, Free)
- **URL**: https://www.who.int/news-room/fact-sheets
- Covers: 100+ diseases — malaria, diabetes, dengue, TB, HIV, cancer, etc.
- License: Open (CC BY-NC-SA 3.0 IGO)
- Download: Right-click → Save as PDF on each fact sheet

### ✅ Option 3: MedlinePlus Medical Encyclopedia
- **URL**: https://medlineplus.gov/encyclopedia.html
- Covers: Diseases, drugs, tests, surgeries, wellness
- License: Public domain (US government)
- Format: HTML printable as PDF

### ✅ Option 4: StatPearls (PubMed Open Access)
- **URL**: https://www.ncbi.nlm.nih.gov/books/NBK430685/
- 5000+ peer-reviewed medical chapters — free PDF downloads
- Ideal for symptoms, diagnosis, treatment of all conditions

### ✅ Option 5: Harrison's Principles (Textbook — if licensed)
- Standard medical school reference
- Check your university library for digital access

### 📁 Recommended Files for `/data` Folder
```
data/
├── who_malaria_factsheet.pdf
├── who_diabetes_factsheet.pdf
├── who_dengue_factsheet.pdf
├── who_tuberculosis_factsheet.pdf
├── who_hiv_aids_factsheet.pdf
├── who_cancer_factsheet.pdf
├── who_hypertension_factsheet.pdf
├── who_nutrition_factsheet.pdf
├── msd_fever_chapter.pdf
├── msd_respiratory_chapter.pdf
└── medlineplus_first_aid.pdf
```

---

## 📁 Project Structure

```
medical-chatbot/
├── app.py                    # Streamlit UI (main entry point)
├── train_index.py            # Build FAISS index from PDFs
├── chatbot.py                # RAG pipeline + answer generation
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/                     # Place your medical PDFs here
│   └── (your PDF files)
│
├── faiss_index/              # Auto-generated after train_index.py
│   ├── index.faiss
│   ├── index.pkl
│   └── index_stats.json
│
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py         # PDF text extraction
│   ├── text_splitter.py      # Chunking logic
│   ├── embeddings.py         # Sentence-transformer embeddings
│   └── vector_store.py       # FAISS build/save/load/search
│
├── evaluation/
│   ├── evaluate.py           # Accuracy + timing evaluation script
│   └── plots/                # Auto-generated evaluation graphs
│
└── assets/                   # Logo / images (optional)
```

---

## 🔧 Technology Stack

| Component         | Library / Model                          |
|-------------------|------------------------------------------|
| PDF Loading       | PyPDF, PDFPlumber                        |
| Text Chunking     | LangChain RecursiveCharacterTextSplitter |
| Embeddings        | sentence-transformers/all-MiniLM-L6-v2   |
| Vector Database   | FAISS (CPU, local)                       |
| Answer Generation | google/flan-t5-base (HuggingFace)        |
| UI                | Streamlit                                |
| Evaluation        | Matplotlib, Pandas, NumPy                |

---

## ⚙️ How It Works (RAG Pipeline)

```
User Question
     │
     ▼
Embed Question (MiniLM-L6-v2)
     │
     ▼
FAISS Similarity Search → Top-K Chunks
     │
     ▼
Build Context from Chunks + Metadata
     │
     ▼
Medical-Safe Prompt Template
     │
     ▼
Flan-T5 Generator → Answer
     │
     ▼
Return: Answer + Sources + Confidence Scores
```

---

## 💻 Installation

### Prerequisites
- Python 3.9 or 3.10
- Windows / macOS / Linux
- VS Code (recommended)
- 4GB RAM minimum (8GB recommended)

### Step 1: Clone / Download the project
```bash
git clone https://github.com/your-repo/medical-chatbot.git
cd medical-chatbot
```

### Step 2: Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ First install may take 5–10 minutes (downloading PyTorch + transformers)

---

## 🚀 Running the Project

### Step 1: Add PDFs to the `data/` folder
Download WHO fact sheets or other medical PDFs and place them in `data/`.

### Step 2: Build the FAISS index
```bash
python train_index.py
```

Optional parameters:
```bash
python train_index.py --chunk_size 800 --overlap 80
python train_index.py --chunk_size 500 --overlap 50
python train_index.py --chunk_size 1000 --overlap 100
```

### Step 3: Start the chatbot
```bash
streamlit run app.py
```
Then open: http://localhost:8501

---

## 📊 Evaluation

Run the evaluation script to generate accuracy graphs:
```bash
python evaluation/evaluate.py
```

Generates:
- `evaluation/plots/response_times.png`
- `evaluation/plots/accuracy_confidence.png`
- `evaluation/plots/chunk_size_comparison.png`
- `evaluation/plots/summary_dashboard.png`
- `evaluation/evaluation_results.csv`

---

## 💬 Example Queries

| Question | Expected Answer Topics |
|----------|----------------------|
| What are the symptoms of malaria? | fever, chills, headache, vomiting, sweating |
| How to prevent diabetes? | diet, exercise, weight control, sugar intake |
| What is dengue fever? | Aedes mosquito, virus, platelet count |
| How does tuberculosis spread? | airborne, coughing, close contact |
| What foods help during a fever? | fluids, hydration, rest, nutrition |

---

## ⚠️ Disclaimer

> This chatbot is for **educational purposes only** and is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.

---

## 🏆 Project Evaluation Metrics

| Metric | Value (typical) |
|--------|----------------|
| Avg Response Time | 1.5 – 4.0 seconds |
| Retrieval Confidence | 0.45 – 0.75 |
| Keyword Accuracy | 55% – 75% |
| Index Build Time | 30 – 120 seconds |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: FAISS index` | Run `python train_index.py` first |
| `No PDFs found` | Add PDF files to the `data/` folder |
| Out of RAM | Use `flan-t5-small` in `chatbot.py` (GENERATOR_MODEL) |
| Slow on first query | Models download from HuggingFace on first run |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

*Built with ❤️ using LangChain · FAISS · HuggingFace · Streamlit*
