# ──────────────────────── rag_pipeline.py ─────────────────────────
from dotenv import load_dotenv
import os, requests, re, time, json
from typing import List
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_community.vectorstores import FAISS
import faiss

# ─── Load Environment ───
load_dotenv()
OR_KEY  = os.getenv("OPENAI_API_KEY")
OR_BASE = os.getenv("OPENAI_API_BASE")
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL_ID = "BAAI/bge-base-en-v1.5"

# ─── Embedding Classes ───
class LocalHFEmbeddings(Embeddings):
    def __init__(self, model_id: str):
        self.model = SentenceTransformer(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        processed = [f"instruction: {t}" for t in texts]
        return self.model.encode(processed, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([f"instruction: {text}"], convert_to_numpy=True)[0].tolist()

class HFHubQueryEmbedder:
    def __init__(self, model_id: str, hf_token: str):
        self.client = InferenceClient(model=model_id, token=hf_token)

    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(f"instruction: {text}")

# ─── FAISS Vector Store ───
def load_vector_store(docs_path="kgpt/data/static_kgp_docs", faiss_path="kgpt/data/faiss_index", model_id=HF_MODEL_ID):
    print("🔍 Initializing local embedding model for documents...")
    embeddings = LocalHFEmbeddings(model_id=model_id)

    if os.path.exists(faiss_path):
        print("⚠️ Existing FAISS index found. Checking dimension...")
        try:
            test_vector = embeddings.embed_query("test")
            dim = len(test_vector)
            index_file = os.path.join(faiss_path, "index.faiss")
            index = faiss.read_index(index_file)
            if dim != index.d:
                raise ValueError(f"Embedding dim mismatch: {dim} vs {index.d}")

            vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            print(f"✅ Loaded FAISS index from: {faiss_path}")
            return vector_store
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            return None

    print("📦 Building FAISS index from scratch...")
    docs = []
    for fname in os.listdir(docs_path):
        full_path = os.path.join(docs_path, fname)
        if fname.endswith(".pdf"):
            loader = UnstructuredPDFLoader(full_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = fname
            docs.extend(loaded_docs)
        elif fname.endswith(".txt"):
            loader = TextLoader(full_path, encoding="utf-8")
            docs.extend(loader.load())
        elif fname.endswith(".json"):
            with open(full_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False, indent=2)
                docs.append(Document(page_content=text, metadata={"source": fname}))

    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = splitter.split_documents(docs)
    print(f"📊 Total chunks: {len(chunks)}")

    vector_store = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
    vector_store.save_local(faiss_path)
    print(f"✅ Saved FAISS index at: {faiss_path}")
    return vector_store

# ─── Wikipedia Fallback ───
def wikipedia_summary(query, lang='en'):
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", query.lower())
    from urllib.parse import quote
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(cleaned.title())}"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json().get("extract")
    return None

# ─── Basic Relevance Filter ───
def is_context_relevant(query, context, threshold=1):
    query_words = set(word.lower() for word in query.split() if len(word) > 3)
    match_count = sum(1 for word in query_words if word in context.lower())
    return match_count >= threshold
