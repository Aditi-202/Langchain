
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

doc1 = Document(
    page_content="virat koli was RCB's captain",
    metadata={"team":"RCB"}
)

doc2 = Document(
    page_content="Dhoni was CSk's captain",
    metadata={"team":"CSk"}
)

doc3 = Document(
    page_content="Rahul was Delhi's captain",
    metadata={"team":"Delhi"}
)

doc4 = Document(
    page_content="Shreyas was Punjab's captain",
    metadata={"team":"Punjab"}
)

docs = [doc1, doc2, doc3, doc4]

vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings,
    persist_directory='chroma_db',
    collection_name='sample'
)

vector_store.add_documents(docs) # add documents
vector_store.get(include=['embeddings', 'documents', 'metadatas']) # view documents

vector_store.similarity_search(
    query='who is captain of RCB',
    k=2,
)