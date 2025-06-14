from kgpt.agent.rag_pipeline import rag_chain
def query_rag(query: str):
    """
    Function to query the RAG chain.
    """
    return rag_chain.invoke({"question": query})
def get_schedule():
    return "IIT KGP schedule is not available yet. Please check back later."
