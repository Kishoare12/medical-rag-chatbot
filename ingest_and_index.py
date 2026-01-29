import os
import glob
from tqdm import tqdm
import chromadb
import torch
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer


# === CONFIGURATION ===
# Use environment variable or default path for development
input_dir = os.getenv("DATA_DIR", r"C:\\Users\\kisho\\OneDrive\\Desktop\\Dataset_chunks")
persist_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
collection_name = "medical_docs"


# === LOAD EMBEDDING MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


# === INITIALIZE CHROMA CLIENT ===
print("⚙️ Initializing ChromaDB...")
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(name=collection_name)
print(f"✅ Connected to Chroma collection: {collection_name}\n")


# === READ CHUNK FILES IN PARALLEL ===
def read_file(file_path):
    """Reads a text file and returns (filename, content)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return os.path.basename(file_path), text
    except Exception as e:
        print(f"⚠️ Skipping {file_path}: {e}")
        return None, None


print("📂 Reading chunk files...")
files = glob.glob(os.path.join(input_dir, "*.txt"))
if not files:
    raise FileNotFoundError(f"No .txt files found in {input_dir}")

# Parallel file reading
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(read_file, files), total=len(files), desc="Reading files"))

# Filter out empty or failed reads
results = [(f, t) for f, t in results if f and t]
print(f"\n✅ Loaded {len(results)} valid chunks out of {len(files)} files.\n")


# === ENCODE AND ADD TO CHROMA IN BATCHES ===
batch_size = 256
print("🚀 Starting embedding + indexing...")

for i in tqdm(range(0, len(results), batch_size), desc="Indexing batches"):
    batch = results[i:i + batch_size]
    ids = [f for f, _ in batch]
    texts = [t for _, t in batch]
    metas = [{"source": f} for f in ids]

    # Encode text batch
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Add to Chroma collection
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metas,
        ids=ids
    )

print("\n✅ All chunks indexed successfully!")
print(f"📦 Data persisted at: {persist_dir}")
