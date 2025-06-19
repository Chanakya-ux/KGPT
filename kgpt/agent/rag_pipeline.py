# ─── 0) Imports and Environment ─────────────────────────────────────────────
from dotenv import load_dotenv
import os, requests, re
import time

load_dotenv()
OR_KEY  = os.getenv("OPENAI_API_KEY")
OR_BASE = os.getenv("OPENAI_API_BASE")

# Hugging Face credentials
HF_TOKEN = "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"
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ─── 1) Loaders and Splitters ───────────────────────────────────────────────
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# ─── 2) FAISS Vector Store ──────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from typing import List
from huggingface_hub import InferenceClient


class HFHubEmbeddings:
    def __init__(self, model_id: str, hf_token: str):
        self.client = InferenceClient(model=model_id, token=hf_token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.client.feature_extraction(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text)

    def __call__(self, text: str) -> List[float]:
        # Support legacy LangChain usage where the object is called directly
        return self.embed_query(text)

def load_vector_store(
    docs_path: str = "kgpt/data/static_kgp_docs",
    faiss_path: str = "kgpt/data/faiss_index",
    model_id: str = HF_MODEL_ID,
    hf_token: str = HF_TOKEN
):
    embeddings = HFHubEmbeddings(model_id=model_id, hf_token=hf_token)

    if os.path.exists(faiss_path):
        print(f"Loading Faiss index from {faiss_path} ...")
        return FAISS.load_local(
            faiss_path, embeddings,
            normalize_L2=True,
            allow_dangerous_deserialization=True
        )
    else:
        print("→ Building FAISS index from scratch...")
        docs = []
        for fname in os.listdir(docs_path):
            if fname.lower().endswith(".pdf"):
                loader = UnstructuredPDFLoader(os.path.join(docs_path, fname))
                docs.extend(loader.load())
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
        vector_store.save_local(faiss_path)
        return vector_store

# ─── 3) Wikipedia Fallback ──────────────────────────────────────────────────
def wikipedia_summary(query, lang='en'):
    alias_map = {
        "subhdip mukherjee": "shubdip mukherjee",
        "subhdeep mukherjee": "shubdip mukherjee"
    }

    original_query = query.lower().strip()
    for typo, corrected in alias_map.items():
        if typo in original_query:
            original_query = corrected
            break

    cleaned = re.sub(r"(who|what|when|where|why|how|is|are|was|were|does|do|did|tell me about|explain|can you)\s+", "", original_query, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s]", "", cleaned).strip()

    words = cleaned.split()
    if len(words) > 4:
        cleaned = " ".join(words[-4:])

    from urllib.parse import quote
    page_title = quote(cleaned.title())

    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    print(f"→ Wikipedia URL: {url}")
    res = requests.get(url)
    print(f"→ Wikipedia status: {res.status_code}")
    if res.status_code == 200:
        return res.json().get("extract")
    return None

# ─── 4) Relevance Filter ────────────────────────────────────────────────────
def is_context_relevant(query, context, threshold=1):
    query_words = set(word.lower() for word in query.split() if len(word) > 3)
    context_lower = context.lower()
    match_count = sum(1 for word in query_words if word in context_lower)
    return match_count >= threshold

# ─── 5) Main Inference Pipeline ─────────────────────────────────────────────
if __name__ == "__main__":
    xy=time.time()
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    query = "Who is Elon Musk"

    # 1. Try FAISS retrieval
    docs = retriever.invoke(query)
    retrieved_texts = [d.page_content for d in docs]
    context = "\n\n".join(retrieved_texts).strip()

    # 2. Check relevance
    if not context or not is_context_relevant(query, context):
        print("→ Retrieved context is irrelevant, trying Wikipedia...")
        wiki_info = wikipedia_summary(query)
        if wiki_info:
            context = wiki_info
            print("→ Wikipedia summary found.")
        else:
            context = "No relevant context found."
            print("→ Wikipedia returned nothing.")

    # 3. Build prompt
    prompt = f"""
You are an expert assistant for IIT Kharagpur students.
Given the following context and question, provide a direct, concise answer.
If the context is not helpful, use your own knowledge or general information, but do not mention the context, its relevance, or apologize.
Do not say things like "Based on the context" or "I don't have information"—just answer the question as best as you can.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}
"""

    # 4. Call OpenRouter API
    url = f"{OR_BASE}/chat/completions"
    print(f"→ Hitting URL: {url}")
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
    print("→ HTTP status:", r.status_code)
    print("→ Response body preview:", repr(r.text)[:200], "…")
    r.raise_for_status()
    data = r.json()

    # 5. Extract and display result
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content") or choice.get("text", "")
    answer = content.strip()

    print("\n=== ANSWER ===\n")
    print(answer, "\n")
    print(time.time() - xy, "seconds elapsed")
