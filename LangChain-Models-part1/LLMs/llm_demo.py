from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model = 'model_name') 

result = llm.invoke("what is capital in India?")

print(result)