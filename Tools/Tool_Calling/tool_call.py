from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

#create tool
@tool
def multiply(a, b) -> int:
    """given 2 numbers a, b this tool returns their product """

print(multiply.invoke({"a": 3, "b":4}))

# tool binding
llm_tools = model.bind_tools([multiply])
llm_tools.invoke("hi how are you")
result = llm_tools.invoke("multiply 3 with 10")

multiply.invoke({"a":3, "b":10})