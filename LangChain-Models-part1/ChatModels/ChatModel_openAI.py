from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model  = ChatOpenAI(model = 'gtp-4',temperature = 0.4 )

result= model.invoke("what is the capital of india")
print(result.content) #why result.content, because we need only the conent which has the answer, as result includes all th extra information 
#about the tokens, the model name etc etc give basically inculdes all the extra information so if we give result.content then only the particular 
#answer gets printed.

