# rag_builder.py
# PURPOSE: Read your Oracle knowledge files
#          Convert them to vectors
#          Store in ChromaDB
# RUN THIS ONCE — like CREATE INDEX in Oracle

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("="*55)
print("  RAG BUILDER — Loading Your Oracle Knowledge")
print("="*55)

# ── STEP 1: Initialize ChromaDB ───────────────────────
# This creates a folder called 'oracle_vectordb'
# Think of it as your Oracle tablespace
# All your knowledge will be stored here
print("\n[1/4] Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./oracle_vectordb")

# Delete existing collection if rebuilding
try:
    chroma_client.delete_collection("oracle_knowledge")
    print("      Existing collection deleted. Rebuilding...")
except:
    pass

# Create collection (like CREATE TABLE in Oracle)
collection = chroma_client.create_collection(
    name="oracle_knowledge",
    metadata={"description": "Srikanth 12yr Oracle DBA Knowledge Base"}
)
print("      ChromaDB collection created!")

# ── STEP 2: Load your knowledge files ─────────────────
# Read all .txt files from oracle_knowledge folder
# Think of this like: SELECT * FROM external_table
print("\n[2/4] Loading your Oracle knowledge files...")

knowledge_folder = "./oracle_knowledge"
documents = []
doc_ids = []
doc_metadata = []

for filename in os.listdir(knowledge_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(knowledge_folder, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        # Split into chunks (like partitioning a large table)
        # Each chunk = one piece of knowledge
        # We split by === which separates topics in our files
        chunks = content.split("===")

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) > 100:  # Only meaningful chunks
                documents.append(chunk)
                doc_ids.append(f"{filename}_{i}")
                doc_metadata.append({
                    "source": filename,
                    "chunk_number": i
                })

print(f"      Loaded {len(documents)} knowledge chunks")
print(f"      From files: {os.listdir(knowledge_folder)}")

# ── STEP 3: Convert text to vectors ───────────────────
# sentence-transformers converts your text to numbers
# Similar meaning = similar numbers
# This is what makes search work by MEANING not keywords
print("\n[3/4] Converting knowledge to vectors...")
print("      Loading embedding model (first time = download ~90MB)...")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# This model runs 100% locally on your laptop
# No internet needed after first download
# No API key needed — completely free

embeddings = embedder.encode(documents).tolist()
print(f"      Converted {len(embeddings)} chunks to vectors")
print(f"      Each vector has {len(embeddings[0])} dimensions")

# ── STEP 4: Store in ChromaDB ─────────────────────────
# Like INSERT INTO your vector database
print("\n[4/4] Storing in ChromaDB vector database...")

collection.add(
    ids=doc_ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=doc_metadata
)

print(f"      Stored {collection.count()} chunks in ChromaDB")
print(f"      Database saved to: ./oracle_vectordb/")

# ── TEST: Verify it works ──────────────────────────────
print("\n" + "="*55)
print("  TESTING YOUR KNOWLEDGE BASE")
print("="*55)

test_questions = [
    "How do I fix ORA-04031?",
    "What SQL checks tablespace usage?",
    "What happened in the listener incident?"
]

for question in test_questions:
    print(f"\nQ: {question}")

    # Convert question to vector
    q_vector = embedder.encode([question]).tolist()

    # Search ChromaDB for similar content
    results = collection.query(
        query_embeddings=q_vector,
        n_results=1
    )

    # Show what was found
    found_text = results['documents'][0][0]
    source = results['metadatas'][0][0]['source']
    print(f"Found in: {source}")
    print(f"Content preview: {found_text[:150]}...")

print("\n" + "="*55)
print("  YOUR ORACLE KNOWLEDGE BASE IS READY!")
print("  ChromaDB is loaded with your 12yr expertise")
print("  Run rag_chat.py next to start chatting!")
print("="*55)