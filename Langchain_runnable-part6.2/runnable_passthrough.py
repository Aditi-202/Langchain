from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template="generate a joke about {topic}"
)

prompt2 = PromptTemplate(
    template="explain the following joke - {text}",
    input_variables=['text']
)

parser = StrOutputParser()

sequence_chain = RunnableSequence(prompt1, model, parser)
#print(sequence_chain.invoke({'topic':'AI'}))

parallel_chain = RunnableParallel({
    'Joke' : RunnablePassthrough(),
    'Explanation' :  RunnableSequence(prompt1, model, parser, prompt2, model , parser)
})

final_chain = RunnableSequence(sequence_chain, parallel_chain)
result=final_chain.invoke({'topic':'MBA'})
print(result)