from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):

    name: str = Field(description="name of the person")
    age:int = Field(gt = 18, description="age of the persone")
    city: str = Field(description="name of the city the peron belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="generate the name, age and city of a  fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

#1. without chain
#prompt = template.invoke({'place':'Indian'})
#result  = model.invoke(prompt)
#final = parser.parse(result.content)
#print(final)

#or
#2. with chain

chain = template | model | parser
result = chain.invoke({'place':'Indian'})
print(result)