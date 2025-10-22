from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    input_variables=['poem'],
    template="write a summary for the poem \n {poem}"
)

parser = StrOutputParser()

loader = TextLoader('poem.txt') #loader is textloader's objext in DocumentLoader, when initializing loader we need to specify the path of the txt doc

docs = loader.load() # this document loader has a function called load which baiscally loads the txt doc, just by calling it loads the txt as doc in memory

chain = prompt1 | model | parser

print(chain.invoke({'poem': docs[0].page_content}))