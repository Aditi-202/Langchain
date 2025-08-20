from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name = 'fact_1', description= "about the topic"),
    ResponseSchema(name = 'fact_2', description= "about the topic"),
    ResponseSchema(name = 'fact_3', description= "about the topic"),

]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= "give 3 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

#1. without chain
prompt = template.invoke({'topic': 'black hole'})
result = model.invoke(prompt)
final = parser.parse(result.content)
print(final)

#or
# 2. with chains
chain = template | model | parser
result = chain.invoke({'topic': "black hole"})
print(result)