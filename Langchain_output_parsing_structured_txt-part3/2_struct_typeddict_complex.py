#annotated: when llm reads the keys from the class dict & sometimes don't understand, we can add guide the llm by attaching a line 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal # literal tells that it should be either this or that

load_dotenv()

model = ChatOpenAI()

#schema
class Review(TypedDict):
    key_theme:Annotated[list[str], "write all the key themes discussed in the review in the list"]
    summary: Annotated[str, "a brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Sentiment must be 'pos' or 'neg'"]
    pros: Annotated[Optional[list[str]], "write all the pros inside the list"]
    cons: Annotated[Optional[list[str]], "write all the cons inside the list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" the hardware is great, but the software feels bloated. there are too many pre installed apps that i can't remove. also the ui looks outdated compared to other brands. hoping for a software update to fix this.""")

print(result)