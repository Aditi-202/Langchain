from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Delhi is capital of India")

print(str(result))

#the text goes into the embedding models then it processes it and then a vector gets generated of 32 demension and then it gets printed as result.