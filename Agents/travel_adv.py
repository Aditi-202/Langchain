from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from ddgs import DDGS
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

# Load model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define a proper tool instead of passing result directly
search_tool = DuckDuckGoSearchRun()

# Load react prompt from hub
prompt = hub.pull("hwchase17/react")

# ✅ Create ReAct agent (use `llm` instead of `model`)
agent = create_react_agent(
    llm=model,       # ← FIXED: must be `llm`
    tools=[search_tool],
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

# Invoke agent
response = agent_executor.invoke({"input": "3 ways to reach Goa from Bangalore"})
print(response["output"])
