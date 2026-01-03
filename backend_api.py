from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import asyncio
import time
import os

app = FastAPI(title="Medical RAG Chatbot API (Async Optimized for 60s)")

# --- Initialize models (lightweight & fast) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- Connect to persistent ChromaDB ---
persist_dir = "chroma_db"
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("medical_docs")


@app.get("/")
def root():
    return {"message": "Medical RAG Chatbot API running. Use POST /query with JSON body."}


# --- Async summarization helper ---
async def summarize_contexts_async(docs, start_time):
    loop = asyncio.get_event_loop()

    async def summarize_one(text):
        if time.time() - start_time > 50:
            return None  # stop if nearing timeout
        text = text.replace("\n", " ").replace("REFERENCES", "").strip()[:400]
        if len(text.split()) < 60:
            return text[:150]
        try:
            # run summarizer in a threadpool to avoid blocking event loop
            summary = await loop.run_in_executor(
                None, lambda: summarizer(text, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
            )
            return summary.strip()
        except Exception:
            return text[:150]

    tasks = [summarize_one(d) for d in docs[:5]]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


@app.post("/query")
async def query_docs(request: Request):
    start_time = time.time()
    try:
        # --- Step 1: Parse request ---
        data = await request.json()
        question = data.get("query", "").strip()
        top_k = int(data.get("top_k", 5))

        if not question:
            return JSONResponse({"error": "Missing 'query' field"}, status_code=400)

        # --- Step 2: Retrieve relevant documents ---
        query_vec = embedder.encode([question], convert_to_numpy=True)[0]
        results = collection.query(query_embeddings=[query_vec.tolist()], n_results=top_k)
        docs = results.get("documents", [[]])[0]

        # --- Step 3: Summarize contexts concurrently ---
        contexts = await summarize_contexts_async(docs, start_time)
        contexts = contexts[:3]
        combined_text = " ".join(contexts)

        # --- Step 4: Generate concise answer ---
        prompt = f"Answer briefly and clearly based only on this context.\n\nContext:\n{combined_text}\n\nQuestion: {question}\n\nAnswer:"
        result = qa_pipeline(prompt, max_length=80, temperature=0.2)[0]["generated_text"]

        # --- Step 5: Postprocess answer ---
        if len(result.split()) > 40:
            result = " ".join(result.split()[:40]) + "..."

        return JSONResponse({
            "answer": result.strip(),
            "contexts": contexts
        }, status_code=200)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
