from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents=[
    "Bangalore is the capital of Karnataka",
    "Paris is capital of France",
    "Delhi is captal of India",
    "Kolata is Captial of West Bengal"
]

vector = embedding.embed_documents(documents)

print(str(vector))