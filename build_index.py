import json
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import re

# your imports here — figure out what you need for:
# PDF extraction, embeddings, FAISS

CHUNK_SIZES = [128, 256, 384]
OVERLAP = 50
PDF_PATH = "data/Data Science from Scratch by Joel Grus.pdf"
STORE_PATH = "store/"

def extract_text_from_pdf(pdf_path: str, start_page: int, end_page: int) -> str:
    """
    Extract raw text from the PDF.
    Return a single long string.
    """
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc.pages(start_page, end_page))

def clean_text(text: str) -> str:
    """
    Remove garbage characters, fix spacing issues,
    normalize whitespace.
    Return cleaned string.
    """
    # Remove non-printable characters
    text = re.sub(r'www\.it-ebooks\.info', '', text)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\x0c', '', text)
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    text = re.sub(r'\n{3,}','\n\n', text)
    text = re.sub(r'[ \t]{2,}',' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int, tokenizer) -> list[str]:
    """
    Split text into chunks of approximately chunk_size tokens
    with overlap tokens of overlap between consecutive chunks.
    Return list of chunk strings.

    Think carefully: are you splitting by tokens or by words?
    Why might that distinction matter here?
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation = False, max_length = None)
    start = 0
    chunks = []
    while start < len(token_ids):
        end = start + chunk_size
        chunk = token_ids[start:end]
        decoded_chunk = tokenizer.decode(chunk)
        chunks.append(decoded_chunk)
        start = end - overlap
    return chunks

def build_and_save_index(chunks: list[str], chunk_size: int, model):
    """
    1. Embed all chunks using model.encode()
    2. Ensure float32
    3. Build FAISS IndexFlatIP with dimension 768
    4. Add vectors to index
    5. Save index to store/faiss_{chunk_size}.index
    6. Save chunks to store/chunks_{chunk_size}.json
    """
    embeddings = model.encode(chunks)
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, f"{STORE_PATH}faiss_{chunk_size}.index")
    with open(f"{STORE_PATH}chunks_{chunk_size}.json", "w") as f:
        json.dump(chunks, f)

if __name__ == "__main__":
    model = SentenceTransformer("all-mpnet-base-v2")
    text = extract_text_from_pdf(PDF_PATH, 12, 322)
    text = clean_text(text)
    
    for size in CHUNK_SIZES:
        chunks = chunk_text(text, chunk_size=size, overlap=OVERLAP, tokenizer = model.tokenizer)
        build_and_save_index(chunks, chunk_size=size,model = model)
        print(f"Built index for chunk size {size}: {len(chunks)} chunks")