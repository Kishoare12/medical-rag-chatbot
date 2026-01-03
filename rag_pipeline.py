import os
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

import openai

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "medical_docs"
CHROMA_DIR = os.getenv('CHROMA_DIR', './chroma_db')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# initialize
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

if OPENAI_KEY:
    openai.api_key = OPENAI_KEY


def retrieve(query: str, top_k: int = 4):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)
    docs = res['documents'][0]
    metadatas = res['metadatas'][0]
    distances = res.get('distances', [[]])[0]
    results = []
    for d, m, dist in zip(docs, metadatas, distances):
        results.append({'text': d, 'metadata': m, 'score': dist})
    return results


def build_prompt(query: str, retrieved: List[dict]) -> str:
    context = "\n\n".join([f"Source: {r['metadata']['source']} | chunk_id: {r['metadata']['chunk_id']}\n{r['text']}" for r in retrieved])
    prompt = f"You are an evidence-first clinical assistant. Use only the context below to answer the question. If the context does not contain the answer, say 'I don't know' and recommend consulting a medical professional.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer concisely and include the sources used."
    return prompt


def generate_answer(prompt: str, model: str = 'gpt-4o-mini', max_tokens: int = 300):
    # If OPENAI_KEY not set, raise an informative error
    if not OPENAI_KEY:
        raise EnvironmentError('OPENAI_API_KEY not set. Set OPENAI_API_KEY to call the LLM, or replace generate_answer with a local model call.')
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{'role':'user','content':prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    return resp['choices'][0]['message']['content']


def answer_query(query: str, top_k: int = 4):
    retrieved = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)
    return {'answer': answer, 'sources': retrieved}