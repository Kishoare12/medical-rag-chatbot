from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Paths
persist_dir = "./chroma_db"

# Load your local Chroma database
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("medical_docs")

# Load the same embedding model used during indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/query")
async def query_docs(request: Request):
    """Handles user question and retrieves top relevant chunks."""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        if not question:
            return JSONResponse({"error": "Missing question"}, status_code=400)

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
