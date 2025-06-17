"""Wrappers for tools used by the LangChain agent."""

from kgpt.agent.rag_pipeline import (
    load_vector_store,
    is_context_relevant,
    wikipedia_summary,
)

_vector_store = load_vector_store()
_retriever = _vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 5}
)


def query_rag(query: str) -> str:
    """Return relevant context for a query using the RAG pipeline."""
    docs = _retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    if not context or not is_context_relevant(query, context):
        wiki_info = wikipedia_summary(query)
        context = wiki_info if wiki_info else "No relevant context found."
    return context
def get_schedule():
    return "IIT KGP schedule is not available yet. Please check back later."
