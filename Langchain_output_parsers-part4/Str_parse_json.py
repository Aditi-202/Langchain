from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template="give me the name , age and city of fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# 1. without chain function
prompt = template.format()
result = model.invoke(prompt)
#print(result)
final = parser.parse(result.content)
print(final)

#or

#2. with chain function
chain = template | model | parser
result= chain.invoke({}) 
print(result)