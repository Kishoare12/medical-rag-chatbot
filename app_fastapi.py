from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

app = FastAPI()

# configure basic logging for easier debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Paths
persist_dir = "./chroma_db"

# Load your local Chroma database
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("medical_docs")

# Load the same embedding model used during indexing
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/")
def root():
    return {"message": "Medical RAG Chatbot API running. Use POST /query with JSON body (keys: 'query' or 'question')."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
async def query_docs(request: Request):
    """Handles user question and retrieves top relevant chunks."""
    try:
        # read JSON body safely and log it for debugging
        try:
            data = await request.json()
        except Exception:
            raw = await request.body()
            logger.info(f"Received non-json body: {raw}")
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)

        logger.info(f"Incoming query payload: {data}")

        # accept either 'query' or 'question' for compatibility with different frontends
        question = (data.get("query") or data.get("question") or "").strip()
        try:
            top_k = int(data.get("top_k", 5))
        except Exception:
            top_k = 5
        if not question:
            return JSONResponse({"error": "Missing 'query' or 'question' field"}, status_code=400)

        # Convert question to embedding
        query_vec = model.encode([question], convert_to_numpy=True)[0]

        # Search in ChromaDB for similar chunks
        results = collection.query(query_embeddings=[query_vec.tolist()], n_results=5)

        # Collect answers
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Combine top results into a single answer
        combined_answer = "\n\n---\n\n".join(docs)

        response = {
            "query": question,
            "answer": combined_answer,
            "sources": [m.get("source", "Unknown") for m in metas]
        }

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
