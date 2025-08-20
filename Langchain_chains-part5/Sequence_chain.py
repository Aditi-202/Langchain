from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="generate 5 facts about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate 3 pointer summary from the folllowing text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 |model | parser
result = chain.invoke({'topic':'Rina Kent'})
print(result)