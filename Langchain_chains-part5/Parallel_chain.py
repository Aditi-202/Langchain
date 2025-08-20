#notes to quiz application
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model1 = ChatHuggingFace(llm = llm)
model2 = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="generate short and simple notes from following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="generate 3 short question answer from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge the provided noted and quiz into a single document  \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)


parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain

text = """
A good neighbor next door makes the street feel safer, kinder, and more alive. They exchange small words that brighten dull days—“Good morning,” “How are you?” or “That plant of yours is blooming beautifully.” These simple gestures weave an invisible thread of connection, reminding us that community is built not just in grand gestures, but in little moments of trust and shared living.

Whether you borrow sugar, celebrate festivals together, or just wave across the yard, the neighbor next door stands as a quiet companion in the everyday story of home.
"""

result = chain.invoke({'text': text})
print(result)