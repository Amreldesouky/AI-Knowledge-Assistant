# 🧠 AI Knowledge Assistant (RAG Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your uploaded PDFs using **Gemini** embeddings + **FAISS** vector search.

---

## 🚀 Quick Setup (3 steps)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

---

## 🔑 Free Gemini API Key

1. Visit https://aistudio.google.com
2. Click **Get API Key** → Create API Key
3. Paste it in the app sidebar

---

## 📖 Architecture

```
PDFs → Extract text → Chunk (300 words) → Gemini Embeddings → FAISS Index
                                                                      ↓
User Question → Embed → Similarity Search → Top 4 Chunks → Gemini LLM → Answer
```

| Component | Tool |
|-----------|------|
| PDF parsing | PyPDF2 |
| Embeddings | Gemini text-embedding-004 |
| Vector search | FAISS (IndexFlatL2) |
| LLM | Gemini 2.5 Flash |
| UI | Streamlit |

---

## ✨ Features

- Upload multiple PDFs
- Semantic similarity search
- Conversation memory (last 3 turns)
- Shows retrieved source chunks
- 100% free (Gemini free tier)

---

## ⚠️ Limitations

- Text-based PDFs only (not scanned images)
- Index resets on page refresh
- Gemini free tier: ~15 requests/min

## 🔧 Improvements

- Persist FAISS index to disk
- Add OCR for scanned PDFs
- Stream answers token by token
- Add page number citations
