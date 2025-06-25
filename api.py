# ─────────────────────── api.py ─────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
from kgpt.agent.rag_pipeline import (
    load_vector_store,
    HFHubQueryEmbedder,
    wikipedia_summary,
    is_context_relevant,
    HF_MODEL_ID,
    HF_TOKEN
)

load_dotenv()
OR_KEY = os.getenv("OPENAI_API_KEY")
OR_BASE = os.getenv("OPENAI_API_BASE")

app = FastAPI(title="KGPT Web Service")

vector_store = None
retriever = None

class QueryInput(BaseModel):
    question: str

@app.on_event("startup")
def load_pipeline():
    global vector_store, retriever
    vector_store = load_vector_store()

    if not vector_store:
        raise RuntimeError("❌ Vector store failed to load.")

    # Inject HF API for query embedding only
    hf_embedder = HFHubQueryEmbedder(model_id=HF_MODEL_ID, hf_token=HF_TOKEN)

    class HybridRetriever:
        def __init__(self, vector_store, embedder):
            self.vector_store = vector_store
            self.embedder = embedder

        def invoke(self, query: str):
            embedding = self.embedder.embed_query(query)
            return self.vector_store.similarity_search_by_vector(embedding, k=15)

    retriever = HybridRetriever(vector_store, hf_embedder)

@app.post("/query")
def query_kgpt(input_data: QueryInput):
    query = input_data.question.strip()
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    if not context or not is_context_relevant(query, context):
        wiki_info = wikipedia_summary(query)
        context = wiki_info if wiki_info else "No relevant context found."

    prompt = f"""
You are an expert assistant for IIT Kharagpur students, possessing comprehensive knowledge about the institution. Your name is \"KGPT\" and were developed by two AI enthusiasts: Chanukya (23EC10056) and Sivaram (23EC10023).

=== CONTEXT ===
{context}

=== QUESTION By IITKGP student ===
{query}

Answer the question using only the context. If the question is about professors, list only professors from the department mentioned. Avoid phrases like \"based on the context\" or \"I don't have that information.\" Use emojis to make the answer fun and helpful.
DO not include things like "These are the professors currently mentioned in the context provided".
"""

    body = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(f"{OR_BASE}/chat/completions", json=body, headers=headers)
    r.raise_for_status()
    response = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    return {"answer": response}
