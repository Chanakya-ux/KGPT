# ──────────────────────── rag_pipeline.py ─────────────────────────
import os
import faiss
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# ─── Load Environment ───
load_dotenv()
HF_MODEL_ID = "BAAI/bge-base-en-v1.5"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# ─── Load FAISS Index Only ───
def load_vector_store(faiss_path="kgpt/data/faiss_index"):
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"❌ FAISS path not found: {faiss_path}")

    index_file = os.path.join(faiss_path, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"❌ FAISS index file not found at: {index_file}")

    print(f"📦 Loading FAISS index from: {faiss_path}")
    vector_store = FAISS.load_local(
        folder_path=faiss_path,
        embeddings=None,  # You’ll inject the embedder later
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded successfully.")
    return vector_store

# ─── Wikipedia Fallback ───
