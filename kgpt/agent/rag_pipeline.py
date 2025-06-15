from dotenv import load_dotenv
import os, requests, re

# ─── 1) Load environment variables ──────────────────────────────────────────
load_dotenv()
OR_KEY  = os.getenv("OPENAI_API_KEY")
OR_BASE = os.getenv("OPENAI_API_BASE")

# ─── 2) Load documents and split ────────────────────────────────────────────
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# ─── 3) Embeddings + FAISS ──────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vector_store(
    docs_path: str = "kgpt/data/static_kgp_docs",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    faiss_path: str = "kgpt/data/faiss_index"
):
    if os.path.exists(faiss_path):
        print(f"Loading Faiss index from {faiss_path} ...")
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
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
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        vector_store = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
        vector_store.save_local(faiss_path)
        return vector_store

# ─── 4) Wikipedia fallback ──────────────────────────────────────────────────
def wikipedia_summary(query, lang='en'):
    import re

    # Known alias corrections
    alias_map = {
        "subhdip mukherjee": "shubdip mukherjee",
        "subhdeep mukherjee": "shubdip mukherjee"
    }

    original_query = query.lower().strip()

    # Apply alias correction if needed
    for typo, corrected in alias_map.items():
        if typo in original_query:
            original_query = corrected
            break

    # Remove common question words
    cleaned = re.sub(r"(who|what|when|where|why|how|is|are|was|were|does|do|did|tell me about|explain|can you)\s+", "", original_query, flags=re.IGNORECASE)

    # Remove trailing punctuation
    cleaned = re.sub(r"[^\w\s]", "", cleaned).strip()

    # Use only last 4 words (to avoid full sentence fragments)
    words = cleaned.split()
    if len(words) > 4:
        cleaned = " ".join(words[-4:])

    page_title = cleaned.replace(" ", "_").capitalize()

    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    print(f"→ Wikipedia URL: {url}")
    res = requests.get(url)
    print(f"→ Wikipedia status: {res.status_code}")
    if res.status_code == 200:
        return res.json().get("extract")
    return None

# ─── 5) Relevance scoring helper ────────────────────────────────────────────
def is_context_relevant(query, context, threshold=1):
    query_words = set(word.lower() for word in query.split() if len(word) > 3)
    context_lower = context.lower()
    match_count = sum(1 for word in query_words if word in context_lower)
    return match_count >= threshold

# ─── 6) Main inference pipeline ─────────────────────────────────────────────
if __name__ == "__main__":
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    query = "explain about quantum mechanics in simple terms"

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
