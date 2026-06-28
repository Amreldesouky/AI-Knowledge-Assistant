# 🧠 AI Knowledge Assistant (RAG Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your uploaded PDFs using **Llama 3.3 70B** via Groq + **FAISS** vector search.

## **Live Demo (https://ai-knowledge-assistant-ejhw9zpbrpudjucxpcghsh.streamlit.app/)**
---

## 🚀 Quick Setup

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push `app.py` and `requirements.txt` to a GitHub repo
2. Go to https://share.streamlit.io
3. Sign in with GitHub → Click **New app**
4. Select your repo → Main file: `app.py` → Click **Deploy**
5. Your app will be live at `https://your-app.streamlit.app`

> Enter your Groq API Key in the sidebar after deployment — no environment variables needed.

---

## 🔑 Free Groq API Key

1. Go to https://console.groq.com
2. Sign up → API Keys → **Create API Key**
3. Paste it in the app sidebar

---

## 📖 Architecture

```
PDFs → Extract text → Chunk (300 words) → TF-IDF Vectors → FAISS Index
                                                                   ↓
User Question → Embed → Similarity Search → Top 4 Chunks → Llama 3.3 70B → Answer
```

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Embeddings | TF-IDF (numpy) |
| Vector search | FAISS (IndexFlatL2) |
| LLM | Llama 3.3 70B via Groq |

---

## ✨ Features

- Upload multiple PDFs
- Automatic chunking & indexing
- Semantic similarity search (FAISS)
- Conversation memory (last 3 turns)
- Shows retrieved source chunks
- 100% free (Groq free tier)
- One-click deploy to Streamlit Cloud

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py            # Full app (single file)
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

## ⚠️ Limitations

- Text-based PDFs only (not scanned/image PDFs)
- Index resets on page refresh
- Groq free tier: ~30 requests/min

## 🔧 Improvements

- Add Sentence Transformers for semantic embeddings
- Persist FAISS index to disk
- Add OCR for scanned PDFs
- Stream responses token by token
- Deploy with FastAPI backend
