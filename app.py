import streamlit as st
import PyPDF2
import numpy as np
import faiss
import io
from groq import Groq

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠", layout="wide")
st.title("🧠 AI Knowledge Assistant")
st.caption("Upload PDFs → Ask questions → Get answers with source context")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password",
                            help="Free key at https://console.groq.com")
    st.markdown("---")
    st.markdown("**خطوات:**\n1. حط الـ API Key\n2. ارفع PDF\n3. اضغط Process\n4. اسأل!")
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

for key, val in [("index", None), ("chunks", []), ("chat_history", []), ("embedder", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# Simple TF-IDF-like embedding using numpy (no external model needed)
def simple_embed(texts: list[str], vocab=None) -> tuple[np.ndarray, dict]:
    if vocab is None:
        vocab = {}
        idx = 0
        for text in texts:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
    
    dim = max(len(vocab), 1)
    dim = min(dim, 2000)  # cap dimension
    vectors = np.zeros((len(texts), dim), dtype="float32")
    
    for i, text in enumerate(texts):
        words = text.lower().split()
        for word in words:
            if word in vocab and vocab[word] < dim:
                vectors[i][vocab[word]] += 1.0
        # normalize
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] = vectors[i] / norm
    
    return vectors, vocab

def extract_text(pdf_file) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, size: int = 300, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 20]

def build_index(chunks: list[str]):
    vectors, vocab = simple_embed(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vocab

def retrieve(query: str, k: int = 4) -> list[str]:
    q_vec, _ = simple_embed([query], vocab=st.session_state.vocab)
    # pad if needed
    idx_dim = st.session_state.index.d
    if q_vec.shape[1] < idx_dim:
        q_vec = np.pad(q_vec, ((0,0),(0, idx_dim - q_vec.shape[1])))
    elif q_vec.shape[1] > idx_dim:
        q_vec = q_vec[:, :idx_dim]
    _, idxs = st.session_state.index.search(q_vec, k)
    return [st.session_state.chunks[i] for i in idxs[0] if i < len(st.session_state.chunks)]

def ask_groq(question: str, context_chunks: list[str]) -> str:
    client = Groq(api_key=api_key)
    context = "\n\n---\n\n".join(context_chunks)
    history = []
    for msg in st.session_state.chat_history[-6:]:
        history.append({"role": msg["role"], "content": msg["content"]})
    
    messages = [
        {"role": "system", "content": f"""You are a helpful assistant. Answer ONLY based on the context below.
If the answer is not in the context, say: "I don't have enough information in the uploaded documents."

CONTEXT:
{context}"""}
    ] + history + [{"role": "user", "content": question}]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# ── Upload & Process ──────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
with col2:
    if uploaded_files:
        st.metric("PDFs uploaded", len(uploaded_files))
    process_clicked = st.button("⚡ Process Documents", type="primary",
                                disabled=not uploaded_files, use_container_width=True)

if process_clicked:
    if not api_key:
        st.error("حط الـ Groq API Key في الـ sidebar!")
    else:
        with st.spinner("جاري المعالجة..."):
            try:
                all_chunks = []
                for f in uploaded_files:
                    text = extract_text(f)
                    all_chunks.extend(chunk_text(text))
                st.session_state.chunks = all_chunks
                index, vocab = build_index(all_chunks)
                st.session_state.index = index
                st.session_state.vocab = vocab
                st.session_state.chat_history = []
                st.success(f"✅ تم! {len(all_chunks)} chunk من {len(uploaded_files)} ملف. ابدأ الأسئلة!")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Chat ──────────────────────────────────────────────────────────────────────
st.markdown("---")

if st.session_state.index is None:
    st.info("👆 ارفع PDF واضغط Process Documents.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📄 السياق اللي اتجاب"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:** {src[:300]}...")

    if question := st.chat_input("اسأل سؤال عن الـ PDFs..."):
        if not api_key:
            st.error("حط الـ API Key!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("بفكر..."):
                    try:
                        sources = retrieve(question)
                        answer = ask_groq(question, sources)
                    except Exception as e:
                        sources = []
                        answer = f"❌ Error: {e}"
                st.write(answer)
                if sources:
                    with st.expander("📄 السياق اللي اتجاب"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i}:** {src[:300]}...")
            st.session_state.chat_history.append({
                "role": "assistant", "content": answer, "sources": sources
            })