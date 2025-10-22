from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template="write a really funny joke about {topic}"
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="explain the following joke - {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model , parser)
print(chain.invoke({'topic':'AI'}))

# here in this logic code only expalnation of joke is printed, because since it's sequence runnable, the joke given by model is gone like not seen to us, only the explaination will be seen, so to see the joke along with explanation we can hence use runnable passthrough.