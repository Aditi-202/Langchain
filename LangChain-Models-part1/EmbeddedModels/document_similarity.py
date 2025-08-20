from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', demensions = 300)

documents = [
    "Bangalore is the capital of Karnataka",
    "Paris is capital of France",
    "Delhi is captal of India",
    "Kolata is Captial of West Bengal",
]

query = "what is capital of France?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0] #[0] because we are converting it to 1D, before it was in 2D

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1][-1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)

#basically this code tells that there are 5 documents where we find embeddings for them
#then we have a question/query which we also an embedding for it
# then we bascially find the closest match b/w the question & documents
# so we are doing that matching using a cosine function bascially it finds the angle b/w the question to all other text present in the documents 
# so which ever has the highest match(thst is the grater cosine angle) that's a match b/w the question & which document.

# we are doing enumerate because we are adding the indexes to the document.
#and then sorting the documents based on the angles with questions/query, lesser angles comes first
#we are doing enumerate because we don't want then documents to get jumbled up after sorthing
#key=lambda x:x[1], meaning is sorting based on the 2nd argument that is doc_embeddings
#[-1] because we get to know the highest score similarity & we also get to know the indexe position of the document.
