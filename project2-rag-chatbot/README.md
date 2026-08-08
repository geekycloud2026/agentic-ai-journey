# Project 2: Oracle DBA RAG Knowledge Chatbot

Built by Srikanth | 12 Years Oracle DBA to Agentic AI Engineer

## What This Does
A RAG chatbot trained on 12 years of Oracle DBA knowledge.
Ask any Oracle question and get answers based on YOUR specific
runbooks, incidents, and procedures — not generic documentation.

## How RAG Works
Your txt files → sentence-transformers → ChromaDB vector database
Your question → search ChromaDB → find relevant chunks → Llama 3 → YOUR answer

## Tech Stack
- LLM: Llama 3.1 via Groq API (FREE)
- Vector DB: ChromaDB (FREE, runs locally)
- Embeddings: sentence-transformers (FREE, runs locally, no API key)
- Framework: LangChain
- Language: Python 3.10

## Project Files
- oracle_knowledge/dba_runbook.txt — ORA- error fixes and daily procedures
- oracle_knowledge/performance_tips.txt — Query tuning and space management
- oracle_knowledge/incident_history.txt — 6 real production incidents
- rag_builder.py — Run ONCE to load knowledge into ChromaDB
- rag_chat.py — Run daily to chat with your knowledge base

## How to Run

Step 1: Install packages
pip install langchain langchain-groq groq chromadb sentence-transformers python-dotenv

Step 2: Add Groq API key
echo 'GROQ_API_KEY=your_key_here' > .env

Step 3: Build knowledge base (run once)
python3 rag_builder.py

Step 4: Start chatting
python3 rag_chat.py

## Example Questions to Ask
- How do I fix ORA-04031?
- What was our worst production incident?
- What SQL checks tablespace usage?
- What is the backup procedure?
- How do I tune a slow query?

## What Makes This Different from ChatGPT
ChatGPT gives generic Oracle answers.
This RAG gives answers from YOUR specific runbooks and incidents.
Your 12 years of production knowledge is now searchable by AI.

## Author
Srikanth | Oracle DBA 12 years transitioning to Agentic AI Engineer
GitHub: geekycloud2026
Project 2 of 3 in 30-day Agentic AI journey