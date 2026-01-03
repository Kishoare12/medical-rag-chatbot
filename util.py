from typing import List, Dict, Any
import re
import math

from pypdf import PdfReader

# Simple text extractor for PDFs
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

# A character-based chunker with overlap
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# Attach metadata helper
def make_metadatas(source: str, chunks: List[str]) -> List[Dict[str, Any]]:
    metas = []
    for i, c in enumerate(chunks):
        metas.append({"source": source, "chunk_id": i})
    return metas