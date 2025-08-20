from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents=[
    "Bangalore is the capital of Karnataka",
    "Paris is capital of France",
    "Delhi is captal of India",
    "Kolata is Captial of West Bengal"
]

result = embedding.embed_documents(documents) #this function is for embedding the documents(multiple words)

print(str(result))

#the text goes into the embedding models then it processes it and then a vector gets generated of 32 demension and then it gets printed as result.