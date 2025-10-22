from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from ddgs import DDGS
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests

load_dotenv()

# Load model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define a proper tool instead of passing result directly
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    this function fetches the current weather data for a given city
    """
    url = f'http://api.weatherstack.com/current?access_key=ADD_YOUR_ACCESS_KEY&query={city}'

    response = requests.get(url)
    return response.json()

# Load react prompt from hub
prompt = hub.pull("hwchase17/react")

# ✅ Create ReAct agent (use `llm` instead of `model`)
agent = create_react_agent(
    llm=model,       # ← FIXED: must be `llm`
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

response = agent_executor.invoke({"input": "find the capital of karnataka, then find it's current weather condition"})
print(response)