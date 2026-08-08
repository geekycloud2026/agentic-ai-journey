# rag_chat.py
# PURPOSE: Chat with YOUR Oracle knowledge
#          using RAG (Retrieval Augmented Generation)
# HOW IT WORKS:
#   You ask question
#   → ChromaDB finds relevant knowledge
#   → Llama 3 reads it and answers
#   → Answer is based on YOUR 12yr experience

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time

load_dotenv()

print("="*55)
print("  ORACLE DBA RAG CHATBOT")
print("  Powered by YOUR knowledge + Llama 3")
print("  Type 'exit' to quit")
print("="*55)

# ── Load ChromaDB (your knowledge base) ───────────────
print("\nLoading your Oracle knowledge base...")
chroma_client = chromadb.PersistentClient(path="./oracle_vectordb")
collection = chroma_client.get_collection("oracle_knowledge")
print(f"Loaded {collection.count()} knowledge chunks")

# ── Load embedding model ───────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ── Load Llama 3 via Groq ──────────────────────────────
print("Connecting to Llama 3 via Groq...")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=400
)
print("Ready! Ask me anything about Oracle.\n")
print("-"*55)

def search_knowledge(question, n_results=3):
    """
    Search ChromaDB for relevant knowledge chunks.
    This is the RETRIEVAL part of RAG.
    Like: SELECT relevant_docs FROM knowledge_base
          ORDER BY similarity DESC FETCH FIRST 3 ROWS;
    """
    # Convert question to vector
    q_vector = embedder.encode([question]).tolist()

    # Search for similar vectors in ChromaDB
    results = collection.query(
        query_embeddings=q_vector,
        n_results=n_results
    )

    # Return the actual text chunks found
    chunks = results['documents'][0]
    sources = [m['source'] for m in results['metadatas'][0]]
    return chunks, sources


def ask_with_rag(question):
    """
    Full RAG pipeline:
    1. Search ChromaDB for relevant knowledge
    2. Build prompt with retrieved knowledge
    3. Send to Llama 3 for answer
    4. Return grounded answer
    """
    # STEP 1: RETRIEVAL — search your knowledge base
    chunks, sources = search_knowledge(question)
    context = "\n\n---\n\n".join(chunks)
    unique_sources = list(set(sources))

    # STEP 2: AUGMENTATION — build the prompt
    # This is the KEY step of RAG
    # We give Llama 3 YOUR knowledge as context
    prompt = f"""You are an Oracle DBA AI assistant
trained on Srikanth's 12 years of production experience.

Use ONLY the following knowledge to answer the question.
If the answer is not in the knowledge base say:
"This is not covered in the knowledge base. 
 Based on general Oracle knowledge: [answer]"

KNOWLEDGE BASE:
{context}

QUESTION: {question}

Give a direct, practical answer.
Include exact SQL commands where relevant.
Keep answer under 200 words."""

    # STEP 3: GENERATION — Llama 3 reads and explains
    response = llm.invoke(prompt)
    return response.content, unique_sources


# ── Main chat loop ─────────────────────────────────────
print("\nHello! I am your Oracle DBA AI Assistant.")
print("I am trained on Srikanth's 12 years of experience.")
print("\nTry asking:")
print("  -> How do I fix ORA-04031?")
print("  -> What SQL checks tablespace usage?")
print("  -> What was the worst incident we had?")
print("  -> How do I tune a slow query?")
print("  -> What is the backup procedure?")
print("-"*55)

while True:
    print()
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("AI: Goodbye! Your Oracle knowledge base is always ready.")
        break

    try:
        print("AI: Searching knowledge base...", end="\r")
        answer, sources = ask_with_rag(user_input)
        print(f"AI: {answer}")
        print(f"\n[Sources: {', '.join(sources)}]")
        time.sleep(2)  # Rate limit protection

    except Exception as e:
        if "rate_limit" in str(e).lower():
            print("AI: Rate limit hit. Waiting 30 seconds...")
            time.sleep(30)
            print("AI: Ready! Please ask again.")
        else:
            print(f"AI: Error — {str(e)}")