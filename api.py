from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from kgpt.agent.rag_pipeline import (
    load_vector_store,
    is_context_relevant,
    wikipedia_summary
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OR_KEY = os.getenv("OPENAI_API_KEY")
OR_BASE = os.getenv("OPENAI_API_BASE")

# Initialize FastAPI app
app = FastAPI(title="KGPT Web Service")

# Global variables
vector_store = None
retriever = None

@app.on_event("startup")
def load_pipeline():
    global vector_store, retriever
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Input model
class QueryInput(BaseModel):
    question: str

# API endpoint
@app.post("/query")
def query_kgpt(input_data: QueryInput):
    query = input_data.question.strip()
    
    # 1. Retrieve documents
    docs = retriever.invoke(query)
    retrieved_texts = [d.page_content for d in docs]
    context = "\n\n".join(retrieved_texts).strip()

    # 2. Relevance check + Wikipedia fallback
    if not context or not is_context_relevant(query, context):
        wiki_info = wikipedia_summary(query)
        context = wiki_info if wiki_info else "No relevant context found."

    # 3. Build prompt
    prompt = f"""
You are an expert assistant for IIT Kharagpur students.
Given the following context and question, provide a direct, concise answer.
If the context is not helpful, use your own knowledge or general information, but do not mention the context(like donot mention "according to the context provided" or simiar things), its relevance, or apologize.
Do not say things like "Based on the context" or "I don't have information"—just answer the question as best as you can.
Keep in mind that you are answering questions related to IIT Kharagpur,you know everything about iitkgp,and only iitkgp
=== CONTEXT ===
{context}

=== QUESTION ===
{query}
"""

    # 4. Call OpenRouter
    url = f"{OR_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.5,
    }

    r = requests.post(url, json=body, headers=headers)
    r.raise_for_status()
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    return {"answer": content.strip()}
