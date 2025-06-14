# kgpt/agent/rag_pipeline.py

from dotenv import load_dotenv
import os, requests

# ─── 1) Load environment variables from root `.env` ─────────────────────────
load_dotenv()
OR_KEY  = os.getenv("OPENAI_API_KEY")    # sk-or-v1-…
OR_BASE = os.getenv("OPENAI_API_BASE")   # https://api.openrouter.ai/v1

# ─── 2) Community PDF loader + text splitter ───────────────────────────────
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# ─── 3) Local embeddings + community FAISS ─────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vector_store(
    docs_path: str = "kgpt/data/static_kgp_docs",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    1) Load all PDFs in `docs_path`
    2) Split into ~1k-token chunks
    3) Embed locally
    4) Build and return FAISS index
    """
    docs = []
    for fname in os.listdir(docs_path):
        if fname.lower().endswith(".pdf"):
            loader = UnstructuredPDFLoader(os.path.join(docs_path, fname))
            docs.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.from_documents(chunks, embeddings, normalize_L2=True)

if __name__ == "__main__":
   
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    """"""


    query = "when is admission to first year ug students?"


    """"""
    docs = retriever.get_relevant_documents(query)

    
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""
Use the following context to answer the user question.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== INSTRUCTIONS ===
You are a helpful assistant designed for iitkgp students. Answer the question based on the context provided.
"""

    # ─── 4) Call OpenRouter chat/completions endpoint correctly ─────────────
    url = f"{OR_BASE}/chat/completions"  # https://api.openrouter.ai/v1/chat/completions
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

    # ─── Extract answer ─────────────────────────────────────────────────────
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content") or choice.get("text", "")
    answer = content.strip()

   
    print("\n=== ANSWER ===\n")
    print(answer, "\n")
    