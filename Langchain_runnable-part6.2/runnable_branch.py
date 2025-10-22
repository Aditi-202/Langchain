from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template="write a report on {topic}"
)

prompt2 = PromptTemplate(
    template="summarize the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain= RunnableSequence(report_gen_chain, branch_chain)
print(final_chain.invoke({'topic': 'India vs pakistan'}))