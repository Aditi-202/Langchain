#annotated: when llm reads the keys from the class dict & sometimes don't understand, we can add guide the llm by attaching a line 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatOpenAI()

#schema
class Review(TypedDict):
    summary: Annotated[str, "a brief summary of the review"]
    sentiiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" the hardware is great, but the software feels bloated. there are too many pre installed apps that i can't remove. also the ui looks outdated compared to other brands. hoping for a software update to fix this.""")

print(result)