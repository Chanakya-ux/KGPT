# ─────────────────────── api.py ─────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from typing import List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import numpy as np
import faiss
HF_SECRET ="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"
bge_client = InferenceClient(model="BAAI/bge-base-en-v1.5",token=HF_SECRET)
miniml_client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2", token=HF_SECRET)
faiss_dir = "faiss_mini"
import faiss
import numpy as np
import os

# Load FAISS index
index_file = os.path.join(os.path.dirname(__file__), "index.faiss")
index_mini = faiss.read_index(index_file)

# Load embedding arrays
corpus_mini = np.load(os.path.join(os.path.dirname(__file__), "corpus_mini.npy"))
corpus_bse = np.load(os.path.join(os.path.dirname(__file__), "corpus_bse.npy"))

# Load the original chunks (text corpus) – assuming saved as .txt
with open(os.path.join(os.path.dirname(__file__), "my_chunks.txt"), "r", encoding="utf-8") as f:
    all_text = f.read()

# Split text into individual chunks
corpus = [chunk.strip() for chunk in all_text.split("chunk") if chunk.strip()]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))
def hybrid_embed_query(query: str,top_k=5,intermediate_k=50):
    # Embed query using MiniLM
    query_mini = miniml_client.feature_extraction(f"instruction: {query}")
    
    # Perform initial retrieval using MiniLM
    index_mini = faiss.read_index(index_file)
    D_mini, I_mini = index_mini.search(np.array([query_mini]), intermediate_k)
    top_50 = [corpus[i] for i in I_mini[0]]
    print(top_50)
    top_50_bse = [corpus_bse[i] for i in I_mini[0]]
    query_bse = bge_client.feature_extraction(f"instruction: {query}")
    similarities = cosine_similarity(query_bse, top_50_bse)[0]
    reranked = sorted(zip(top_50, similarities), key=lambda x: x[1], reverse=True)
    top_k_results = reranked[:top_k]
    return top_k_results
from kgpt.agent.rag_pipeline import (
    load_vector_store,
    
    
    HF_MODEL_ID,
    HF_TOKEN
)

load_dotenv()
OR_KEY = os.getenv("OPENAI_API_KEY")
OR_BASE = os.getenv("OPENAI_API_BASE")
class HFHubQueryEmbedder:
    def __init__(self, model_id: str, hf_token: str):
        self.client = InferenceClient(model=model_id, token=hf_token)

    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(f"instruction: {text}")

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
    top_chunks = hybrid_embed_query(query)

    context = "\n\n".join([chunk for chunk in top_chunks])

   

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
