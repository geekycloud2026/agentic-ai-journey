import warnings
warnings.filterwarnings("ignore")
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
import re
import json
import chromadb
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=250
)

print("Loading RAG knowledge base...")
chroma_client = chromadb.PersistentClient(
    path="../project2-rag-chatbot/oracle_vectordb"
)
collection = chroma_client.get_collection("oracle_knowledge")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Loaded {collection.count()} knowledge chunks")

def scanner_agent(log_path: str) -> dict:
    print("\n" + "="*50)
    print("AGENT 1