from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatMessagePromptTemplate([
    ('system','you are a heplful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke({'chay_history':chat_history, 'query':'where is my refund'})
print(prompt)