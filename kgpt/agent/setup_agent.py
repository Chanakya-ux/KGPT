import os
from dotenv import load_dotenv
from langchain_community.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from kgpt.agent.tool_wrappers import get_schedule, query_rag

def get_llm(model_name: str):
    """
    Function to get the LLM based on the model name.
    """
    load_dotenv()
    return ChatOpenAI(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        temperature=0.5
    )

def setup_agent():
    tools = [
        Tool(
            name="Schedule Tool",
            func=get_schedule,
            description="Fetch IITKGP schedule"
        ),
        Tool(
            name="RAG Tool",
            func=query_rag,
            description="Answer from IIT KGP knowledge base"
        )
    ]
    llm = get_llm(model_name="mistralai/mistral-7b-instruct")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    return agent
