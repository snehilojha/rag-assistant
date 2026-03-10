from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from retriever import retrieve
from llm import ask_gemini

class Question(BaseModel):
    question: str

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["model"] = SentenceTransformer("all-mpnet-base-v2")
    yield
    app_state.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(question: Question):
    model = app_state["model"]
    chunks = retrieve(question.question, 384, model, k=5)
    answer = ask_gemini(question.question, chunks)
    return {"answer": answer, "chunks": chunks}
