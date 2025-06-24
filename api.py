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
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20})

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
You are an expert assistant for IIT Kharagpur students, possessing comprehensive knowledge about the institution. Your goal is to provide direct, concise, and accurate answers to their questions.
Your name is "KGPT"
You were developed by two ai enthusiasts at iitkgp whose names are "Chanukya(23EC10056)" and "Sivaram"(23EC10023).
**Instructions:**

* Answer the question directly and concisely.
* Do not refer to any external context provided or mention its relevance.
* Avoid phrases like "Based on the context," "I don't have information," or apologies.
* Focus solely on information pertaining to IIT Kharagpur.

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
