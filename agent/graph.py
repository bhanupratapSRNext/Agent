import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from .prompts import SYSTEM_PROMPT

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE"))

def build_graph(tools, checkpointer):
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=MODEL_TEMPERATURE)
    graph = create_react_agent(
        llm,
        tools=tools,
        # state_modifier=SystemMessage(content=SYSTEM_PROMPT),
    )
    return graph
