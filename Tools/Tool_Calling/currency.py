from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import requests

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

@tool
def get_conversion(base_currency: str, target_currency: str) -> float:
    """This function fetches the currency factor between base and taget currency"""
    url = f'https://v6.exchangerate-api.com/v6/YOUR-API-KEY/pair/{base_currency}/{target_currency}'

    response = requests.get(url)
    return response.json()

get_conversion.invoke({"base_currency":'USD', "target_currency":'INR'})
print(get_conversion)