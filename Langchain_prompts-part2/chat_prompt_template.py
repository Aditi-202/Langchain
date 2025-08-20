from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful {domain} expert'),
    ('human', 'explain in simple terms what is {topic}')
])

prompt = chat_template.invoke({'domain': 'travel', 'topic':'places'})

print(prompt)