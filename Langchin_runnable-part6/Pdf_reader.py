from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

load_dotenv()

#step1: loading the documents
loader = TextLoader("C:/Users/Aditi RN/OneDrive/Desktop/doc.txt")
documents = loader.load()

#step2: split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

#step3: convert text into embedding & stor into FAISS
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embedding)

#step4: create retriever
retriever = vector_store.as_retriever()

#step5: manually retrieve relevant docs
query = "what is the key take away from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

#step6: combine retrieved text into single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs ])

#step7: initialize llm
llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

#step8: manually pass retrieved text to llm
prompt = f"based on the following text, answer the question:{query}\n\n{retrieved_text}"
answer = model.predict(prompt)

#step9: print the answer
print("answer is:", answer)


