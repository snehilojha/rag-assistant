from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from retriever import retrieve, load_all_indices
from ingest import ingest_pdf
from llm import ask_llm
import tempfile
import os
import json
import shutil

import openai


# Request model


class Question(BaseModel):
    question: str
    sources: Optional[list[str]] = None



# App state


app_state = {}



# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app_state["model"] = SentenceTransformer("BAAI/bge-base-en-v1.5")
    app_state["indices"] = load_all_indices("store")
    print(f"Loaded {len(app_state['indices'])} book(s).")
    yield
    print("Shutting down...")



# App

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes

@app.get("/books")
async def get_books():
    """Return metadata for all indexed books."""
    return [
        data["meta"]
        for data in app_state["indices"].values()
    ]

@app.post("/ingest")
async def ingest_book(
    file: UploadFile  = File(...),
    name: str = Form(...),
    author: str = Form(...),):
    """
    Upload a PDF, ingest it into the RAG store, and reload in-memory indices.
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="Book title cannot be empty.")

    if not author.strip():
        raise HTTPException(status_code=400, detail="Author cannot be empty.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    temp_path = None

    try:
        suffix = os.path.splitext(file.filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        meta = ingest_pdf(temp_path, name, author)

        app_state["indices"] = load_all_indices("store")

        return {
            "message": "PDF uploaded and indexed successfully.",
            "title" : meta["title"],
            "author": meta['author'],
            "slug" : meta["slug"],
            "total_chunks": meta["total_chunks"],
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.delete("/books/{slug}")
async def delete_book(slug: str):
    """Delete a stored book by slug, refresh the registry, and reload indices."""
    if not slug.strip():
        raise HTTPException(status_code=400, detail="Slug cannot be empty.")

    base_path = os.path.join("store", slug)
    registry_path = os.path.join("store", "registry.json")

    if not os.path.isdir(base_path):
        raise HTTPException(status_code=404, detail=f"Book '{slug}' not found.")

    try:
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        else:
            registry = []

        registry = [entry for entry in registry if entry.get("slug") != slug]

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

        shutil.rmtree(base_path)

        app_state["indices"] = load_all_indices("store")

        return {"message": "Book deleted successfully.", "slug": slug}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.post("/ask")
async def ask(question: Question):
    """
    Retrieve relevant chunks and generate a grounded answer.

    Error handling:
    - 400: bad input (empty question, unknown source slug)
    - 500: internal retrieval failure
    - 502: OpenAI API failure
    """
    if not question.question or not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    model = app_state["model"]
    indices = app_state["indices"]

    # Retrieval
    try:
        chunks = retrieve(
            question.question,
            model,
            indices=indices,
            sources=question.sources,
            top_k=5
        )
    except ValueError as e:
        # unknown source slug passed in sources
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    # Generation
    try:
        answer = ask_llm(question.question, chunks)
    except ValueError as e:
        # empty query or no chunks
        raise HTTPException(status_code=400, detail=str(e))
    except openai.APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Network error reaching OpenAI: {str(e)}")
    except openai.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e.status_code} — {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    return {"answer": answer, "chunks": chunks}