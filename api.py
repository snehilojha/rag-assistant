from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from retriever import retrieve, load_all_indices
from llm import ask_llm

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