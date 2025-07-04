from fastapi import FastAPI, Request
from pydantic import BaseModel
from kgpt.agent.rag_pipeline import load_vector_store, is_context_relevant, wikipedia_summary

app = FastAPI()

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    query = request.query
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    if not context or not is_context_relevant(query, context):
        wiki_info = wikipedia_summary(query)
        context = wiki_info if wiki_info else "No relevant context found."
    # Build your prompt and call LLM as before...
    # response = call_llm(prompt)
    # return {"answer": response}
    return {"context": context}