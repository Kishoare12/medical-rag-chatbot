import os
from pypdf import PdfReader
from tqdm import tqdm

def extract_text_from_pdf(path):
    text_parts = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def chunk_text(text, chunk_size=1000, overlap=200):
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap

def chunk_folder(input_dir, output_dir, chunk_size=1000, overlap=200):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # show progress bar across all files
    for fname in tqdm(files, desc="Chunking files", unit="file"):
        path = os.path.join(input_dir, fname)
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif fname.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            continue

        base = os.path.splitext(fname)[0]
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            out_file = os.path.join(output_dir, f"{base}_chunk{i}.txt")
            with open(out_file, "w", encoding="utf-8") as out:
                out.write(chunk)

    print(f"\n✅ Finished chunking. Output folder: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    input_dir = r"C:\\Users\\kisho\\OneDrive\\Desktop\\Dataset"
    output_dir = r"C:\|Users\\kisho\\OneDrive\\Desktop\\Dataset_chunks"
    chunk_folder(input_dir, output_dir, chunk_size=1000, overlap=200)
