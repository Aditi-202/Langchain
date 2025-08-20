from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

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

#step7: initialize llm
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational"
    )
)

#create retrievalQAChain
qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, chain_type="stuff")

#ask question
query = "what are key take away from the document?"
answer = qa_chain.invoke({"query": query})
print("answer is:", answer)