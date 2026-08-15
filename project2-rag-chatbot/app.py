# app.py — Project 2 Streamlit Web UI
# RAG DBA Knowledge Chatbot

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time

load_dotenv()

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Oracle DBA AI Assistant",
    page_icon="🗄️",
    layout="wide"
)

# ── Header ─────────────────────────────────────────
st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #0D1B2A, #1B4F8A);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.source-badge {
    background: #EBF5FB;
    border-left: 3px solid #2D9CDB;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    margin-top: 10px;
    color: #1B4F8A;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1 style="color:white; margin:0;">🗄️ Oracle DBA AI Assistant</h1>
    <p style="color:#AED6F1; margin:5px 0 0 0;">
        Powered by Srikanth's 12 Years of Oracle DBA Knowledge + Llama 3
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load RAG components (cached) ───────────────────
@st.cache_resource
def load_rag():
    """Load ChromaDB and embedder once — like connection pooling"""
    with st.spinner("Loading Oracle knowledge base..."):
        client = chromadb.PersistentClient(path="./oracle_vectordb")
        collection = client.get_collection("oracle_knowledge")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=400
        )
    return collection, embedder, llm

collection, embedder, llm = load_rag()

# ── RAG function ───────────────────────────────────
def ask_rag(question):
    """Full RAG pipeline — search + generate"""
    # Search ChromaDB
    q_vector = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=q_vector,
        n_results=3
    )
    chunks = results["documents"][0]
    sources = list(set([
        m["source"] for m in results["metadatas"][0]
    ]))
    context = "\n\n---\n\n".join(chunks)

    # Build prompt
    prompt = f"""You are an Oracle DBA AI assistant trained on
Srikanth's 12 years of production experience.

Use ONLY the following knowledge base to answer.
If not covered say so clearly.

KNOWLEDGE BASE:
{context}

QUESTION: {question}

Give a direct, practical answer with exact SQL where relevant.
Keep under 200 words."""

    # Get answer from Llama 3
    response = llm.invoke(prompt)
    return response.content, sources

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.success("✅ ChromaDB loaded")
    st.info(f"📄 {collection.count()} knowledge chunks")
    st.divider()
    st.markdown("**Topics covered:**")
    st.markdown("• ORA- error fixes")
    st.markdown("• Performance tuning SQL")
    st.markdown("• Backup procedures")
    st.markdown("• Production incidents")
    st.markdown("• Daily health checks")
    st.divider()
    st.markdown("**Try asking:**")
    st.markdown("• How to fix ORA-04031?")
    st.markdown("• What was our worst incident?")
    st.markdown("• What SQL checks tablespace?")
    st.markdown("• How to tune a slow query?")
    st.markdown("• What is the backup procedure?")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat interface ─────────────────────────────────
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am your Oracle DBA AI Assistant. "
                   "I am trained on Srikanth's 12 years of "
                   "production Oracle experience. Ask me anything "
                   "about Oracle errors, performance, backups, "
                   "or incidents!",
        "sources": []
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown(
                f'<div class="source-badge">📄 Sources: '
                f'{", ".join(msg["sources"])}</div>',
                unsafe_allow_html=True
            )

# Chat input
if question := st.chat_input(
        "Ask anything about Oracle DB..."):
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": []
    })
    with st.chat_message("user"):
        st.write(question)

    # Get AI answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                answer, sources = ask_rag(question)
                st.write(answer)
                if sources:
                    st.markdown(
                        f'<div class="source-badge">📄 Sources: '
                        f'{", ".join(sources)}</div>',
                        unsafe_allow_html=True
                    )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                time.sleep(2)
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    st.warning(
                        "Rate limit hit. Wait 30 sec and ask again.")
                else:
                    st.error(f"Error: {str(e)}")