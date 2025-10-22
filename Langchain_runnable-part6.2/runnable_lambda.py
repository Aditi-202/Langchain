from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())


llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template="generate a small joke about {topic}"
)

parser = StrOutputParser()

sequence_chain = RunnableSequence(prompt1, model, parser)
#print(sequence_chain.invoke({'topic':'AI'}))

parallel_chain = RunnableParallel({
    'Joke' : RunnablePassthrough(),
    'word_count' :  RunnableLambda(word_count)
})

final_chain = RunnableSequence(sequence_chain, parallel_chain)
result=final_chain.invoke({'topic':'MBA'})
#print(result)
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])
print(final_result)
